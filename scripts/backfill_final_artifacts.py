from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from radiology_triage.data.chestmnist import build_anomaly_loaders, build_chestmnist_loaders, build_transforms
from radiology_triage.data.multimodal import Vocabulary, build_multimodal_loaders
from radiology_triage.models.autoencoder import ConvAutoencoder
from radiology_triage.models.multimodal import build_multimodal_model
from radiology_triage.models.supervised import build_supervised_model
from radiology_triage.training.multimodal import _forward as multimodal_forward
from radiology_triage.utils.io import load_checkpoint, save_json
from radiology_triage.utils.metrics import compute_binary_confusion_counts, compute_multilabel_confusion_counts
from radiology_triage.utils.plotting import save_binary_confusion_matrix, save_multilabel_confusion_grid


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    backfill_supervised(ROOT / "artifacts" / "supervised" / "simple_cnn" / "best_model.pt")
    backfill_supervised(ROOT / "artifacts" / "supervised" / "resnet18" / "best_model.pt")
    backfill_supervised(ROOT / "artifacts" / "supervised" / "tiny_vit" / "best_model.pt")
    backfill_anomaly(ROOT / "artifacts" / "anomaly" / "conv_autoencoder" / "best_autoencoder.pt")
    backfill_multimodal(ROOT / "artifacts" / "multimodal" / "iu_xray_image_only" / "best_multimodal_model.pt")
    backfill_multimodal(ROOT / "artifacts" / "multimodal" / "iu_xray_text_only" / "best_multimodal_model.pt")
    backfill_multimodal(ROOT / "artifacts" / "multimodal" / "iu_xray_fusion" / "best_multimodal_model.pt")


def backfill_supervised(checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        return
    checkpoint = load_checkpoint(checkpoint_path, map_location=DEVICE)
    config = checkpoint["config"]
    loaders = build_chestmnist_loaders(config)
    model = build_supervised_model(
        model_name=checkpoint["model_name"],
        num_labels=len(checkpoint["label_names"]),
        pretrained=False,
        dropout=checkpoint.get("dropout", config["model"].get("dropout", 0.2)),
        in_channels=3 if config["dataset"].get("as_rgb", True) else 1,
        image_size=checkpoint["image_size"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE).eval()

    labels, probabilities = evaluate_supervised(model, loaders.test)
    confusion = compute_multilabel_confusion_counts(labels, probabilities, label_names=checkpoint["label_names"])
    output_dir = checkpoint_path.parent
    save_json(confusion, output_dir / "test_confusion_counts.json")
    save_multilabel_confusion_grid(
        confusion,
        output_dir / "confusion_matrix.png",
        title=f"Confusion matrices - {checkpoint['model_name']}",
    )


def backfill_anomaly(checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        return
    checkpoint = load_checkpoint(checkpoint_path, map_location=DEVICE)
    config = checkpoint["config"]
    loaders = build_anomaly_loaders(config)
    model = ConvAutoencoder(
        in_channels=checkpoint.get("in_channels", 1),
        latent_channels=config["model"].get("latent_channels", 128),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE).eval()

    scores, targets = evaluate_anomaly(model, loaders.test)
    confusion = compute_binary_confusion_counts(targets, scores, threshold=checkpoint["threshold"])
    output_dir = checkpoint_path.parent
    save_json(confusion, output_dir / "test_confusion_counts.json")
    save_binary_confusion_matrix(confusion, output_dir / "confusion_matrix.png", title="Anomaly confusion matrix")


def backfill_multimodal(checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        return
    checkpoint = load_checkpoint(checkpoint_path, map_location=DEVICE)
    config = checkpoint["config"]
    loaders = build_multimodal_loaders(config)
    model = build_multimodal_model(config, vocab_size=checkpoint["vocab_size"], num_labels=len(checkpoint["label_names"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE).eval()

    mode = config["model"]["mode"].lower()
    labels, probabilities = evaluate_multimodal(model, loaders.test, mode)
    confusion = compute_multilabel_confusion_counts(labels, probabilities, label_names=checkpoint["label_names"])
    output_dir = checkpoint_path.parent
    save_json(confusion, output_dir / "test_confusion_counts.json")
    save_multilabel_confusion_grid(
        confusion,
        output_dir / "confusion_matrix.png",
        title=f"Confusion matrices - {mode}",
    )


@torch.no_grad()
def evaluate_supervised(model: nn.Module, loader: torch.utils.data.DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    all_labels = []
    all_probabilities = []
    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images)
        probabilities = torch.sigmoid(logits).cpu()
        all_labels.append(labels.cpu())
        all_probabilities.append(probabilities)
    return torch.cat(all_labels), torch.cat(all_probabilities)


@torch.no_grad()
def evaluate_anomaly(model: nn.Module, loader: torch.utils.data.DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    all_scores = []
    all_targets = []
    for images, anomaly_labels, _ in loader:
        images = images.to(DEVICE)
        reconstructions = model(images)
        errors = torch.mean(torch.abs(reconstructions - images), dim=(1, 2, 3))
        all_scores.append(errors.cpu())
        all_targets.append(anomaly_labels.cpu())
    return torch.cat(all_scores), torch.cat(all_targets)


@torch.no_grad()
def evaluate_multimodal(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_labels = []
    all_probabilities = []
    for batch in loader:
        labels = batch["labels"].cpu()
        logits = multimodal_forward(model, batch, DEVICE, mode)
        probabilities = torch.sigmoid(logits).cpu()
        all_labels.append(labels)
        all_probabilities.append(probabilities)
    return torch.cat(all_labels), torch.cat(all_probabilities)


if __name__ == "__main__":
    main()
