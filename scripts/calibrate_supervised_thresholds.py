from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from radiology_triage.data.chestmnist import build_chestmnist_loaders  # noqa: E402
from radiology_triage.models.supervised import build_supervised_model  # noqa: E402
from radiology_triage.utils.io import load_checkpoint, save_json  # noqa: E402
from radiology_triage.utils.metrics import calibrate_multilabel_thresholds, compute_multilabel_metrics  # noqa: E402
from radiology_triage.utils.repro import get_device, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate class-wise thresholds for a supervised ChestMNIST model.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a trained supervised checkpoint.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path. Defaults next to the checkpoint.")
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=0.5,
        help="Fallback threshold used as calibration baseline.",
    )
    return parser.parse_args()


def _resolve_checkpoint_path(candidate: Path | None) -> Path:
    if candidate is not None:
        return candidate.resolve()

    manifest_path = ROOT / "artifacts" / "deployment_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest.get("deployed_models", []):
            if entry.get("component", "").lower().startswith("supervision"):
                return (ROOT / entry["checkpoint_path"]).resolve()

    best_path: Path | None = None
    best_score = float("-inf")
    for metrics_path in (ROOT / "artifacts" / "supervised").glob("*/test_metrics.json"):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        score = float(metrics.get("macro_roc_auc", float("-inf")))
        if score > best_score:
            best_score = score
            best_path = metrics_path.parent / "best_model.pt"

    if best_path is None:
        raise FileNotFoundError("No supervised checkpoint could be resolved automatically.")
    return best_path.resolve()


@torch.no_grad()
def _collect_outputs(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_probabilities = []
    all_labels = []
    for images, labels in tqdm(loader, desc="Collect", leave=False):
        logits = model(images.to(device))
        probabilities = torch.sigmoid(logits)
        all_probabilities.append(probabilities.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_labels), torch.cat(all_probabilities)


def main() -> None:
    args = parse_args()
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    seed_everything(config.get("seed", 42))
    device = get_device()

    loaders = build_chestmnist_loaders(config)
    label_names = checkpoint["label_names"]
    model = build_supervised_model(
        model_name=checkpoint["model_name"],
        num_labels=len(label_names),
        pretrained=False,
        dropout=checkpoint.get("dropout", config["model"].get("dropout", 0.2)),
        in_channels=3 if config["dataset"].get("as_rgb", True) else 1,
        image_size=checkpoint["image_size"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    val_labels, val_probabilities = _collect_outputs(model, loaders.val, device)
    test_labels, test_probabilities = _collect_outputs(model, loaders.test, device)

    calibration = calibrate_multilabel_thresholds(
        val_labels,
        val_probabilities,
        label_names=label_names,
        default_threshold=args.default_threshold,
    )
    test_default = compute_multilabel_metrics(
        test_labels,
        test_probabilities,
        label_names=label_names,
        threshold=args.default_threshold,
    )
    test_calibrated = compute_multilabel_metrics(
        test_labels,
        test_probabilities,
        label_names=label_names,
        thresholds=calibration["ordered_thresholds"],
    )

    payload: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "model_name": checkpoint["model_name"],
        "dataset": "ChestMNIST",
        "image_size": int(checkpoint["image_size"]),
        "source_split": "validation",
        "label_names": label_names,
        "default_threshold": float(args.default_threshold),
        "strategy": calibration["strategy"],
        "thresholds": calibration["thresholds"],
        "ordered_thresholds": calibration["ordered_thresholds"],
        "validation_summary": {
            "macro_f1_default": calibration["validation_metrics_default"]["macro_f1"],
            "macro_f1_calibrated": calibration["validation_metrics_calibrated"]["macro_f1"],
            "macro_f1_gain": calibration["macro_f1_gain"],
        },
        "metrics_validation_default": calibration["validation_metrics_default"],
        "metrics_validation_calibrated": calibration["validation_metrics_calibrated"],
        "metrics_test_default": test_default,
        "metrics_test_calibrated": test_calibrated,
        "per_class": calibration["per_class"],
    }

    output_path = args.output.resolve() if args.output is not None else checkpoint_path.with_name("class_thresholds.json")
    save_json(payload, output_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
