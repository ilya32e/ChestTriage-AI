from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from radiology_triage.config import flatten_dict, save_yaml
from radiology_triage.data.multimodal import build_multimodal_loaders
from radiology_triage.models.multimodal import FusionClassifier, build_multimodal_model
from radiology_triage.utils.io import ensure_dir, save_checkpoint, save_json
from radiology_triage.utils.metrics import compute_multilabel_confusion_counts, compute_multilabel_metrics
from radiology_triage.utils.mlflow_utils import setup_mlflow
from radiology_triage.utils.plotting import (
    save_multilabel_confusion_grid,
    save_per_class_metric_plot,
    save_training_curves,
)
from radiology_triage.utils.repro import get_device, seed_everything


def run_multimodal_experiment(config: dict[str, Any]) -> dict[str, Any]:
    seed_everything(config.get("seed", 42))
    device = get_device()
    output_dir = ensure_dir(config["output_dir"])
    save_yaml(config, output_dir / "resolved_config.yaml")

    loaders = build_multimodal_loaders(config)
    model = build_multimodal_model(config, vocab_size=loaders.vocab.size, num_labels=len(loaders.label_names)).to(device)

    training_cfg = config["training"]
    train_labels = torch.tensor(
        loaders.train.dataset.dataframe[loaders.label_names].values,
        dtype=torch.float32,
    )
    positives = train_labels.sum(dim=0)
    negatives = train_labels.shape[0] - positives
    pos_weight = negatives / (positives + 1e-6) if training_cfg.get("use_pos_weight", True) else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["lr"],
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )
    scheduler = _build_scheduler(optimizer, training_cfg)

    mode = config["model"]["mode"].lower()
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_roc_auc": [],
        "val_macro_f1": [],
    }
    best_score = float("-inf")
    best_epoch = 0
    best_checkpoint_path = Path(output_dir) / "best_multimodal_model.pt"
    patience = training_cfg.get("early_stopping_patience", 5)
    patience_counter = 0

    setup_mlflow(config["tracking_uri"], config["experiment_name"])

    run_name = config.get("run_name") or mode
    with mlflow.start_run(run_name=run_name) as active_run:
        run_id = active_run.info.run_id
        mlflow.log_params(flatten_dict(config))
        mlflow.log_param("device", str(device))
        mlflow.log_param("vocab_size", loaders.vocab.size)

        for epoch in range(1, training_cfg["epochs"] + 1):
            train_loss = _train_one_epoch(model, loaders.train, optimizer, criterion, device, mode, epoch)
            val_loss, val_metrics, _ = _evaluate(model, loaders.val, criterion, device, loaders.label_names, mode)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_macro_roc_auc"].append(val_metrics["macro_roc_auc"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_macro_roc_auc": val_metrics["macro_roc_auc"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_macro_average_precision": val_metrics["macro_average_precision"],
                },
                step=epoch,
            )

            if scheduler is not None:
                scheduler.step()

            if val_metrics["macro_roc_auc"] > best_score:
                best_score = val_metrics["macro_roc_auc"]
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(
                    best_checkpoint_path,
                    {
                        "state_dict": model.state_dict(),
                        "config": config,
                        "label_names": loaders.label_names,
                        "vocab_state": loaders.vocab.state_dict(),
                        "vocab_size": loaders.vocab.size,
                        "image_size": config["dataset"]["image_size"],
                        "max_length": config["dataset"]["max_length"],
                        "mlflow_run_id": run_id,
                        "mlflow_experiment_name": config["experiment_name"],
                        "mlflow_run_name": run_name,
                    },
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        test_loss, test_metrics, test_outputs = _evaluate(
            model,
            loaders.test,
            criterion,
            device,
            loaders.label_names,
            mode,
            return_outputs=True,
        )
        if test_outputs is None:
            raise RuntimeError("Expected test outputs for confusion matrix generation.")
        test_confusion = compute_multilabel_confusion_counts(
            test_outputs["labels"],
            test_outputs["probabilities"],
            label_names=loaders.label_names,
        )
        outputs = {
            "best_epoch": best_epoch,
            "best_val_macro_roc_auc": best_score,
            "test_loss": test_loss,
            "test_metrics": test_metrics,
            "checkpoint_path": str(best_checkpoint_path),
            "mlflow_run_id": run_id,
        }

        if mode in {"fusion", "multimodal"} and isinstance(model, FusionClassifier):
            _, text_missing_metrics, _ = _evaluate(
                model,
                loaders.test,
                criterion,
                device,
                loaders.label_names,
                mode,
                disable_text=True,
            )
            _, image_missing_metrics, _ = _evaluate(
                model,
                loaders.test,
                criterion,
                device,
                loaders.label_names,
                mode,
                disable_image=True,
            )
            outputs["robustness"] = {
                "text_missing": text_missing_metrics,
                "image_missing": image_missing_metrics,
            }
            save_json(outputs["robustness"], output_dir / "robustness_metrics.json")
            mlflow.log_metric("test_text_missing_macro_roc_auc", text_missing_metrics["macro_roc_auc"])
            mlflow.log_metric("test_image_missing_macro_roc_auc", image_missing_metrics["macro_roc_auc"])

        save_json(test_metrics, output_dir / "test_metrics.json")
        save_json(test_confusion, output_dir / "test_confusion_counts.json")
        save_training_curves(history, output_dir / "training_curves.png")
        save_per_class_metric_plot(test_metrics["per_class"], "roc_auc", output_dir / "per_class_roc_auc.png")
        save_multilabel_confusion_grid(
            test_confusion,
            output_dir / "confusion_matrix.png",
            title=f"Confusion matrices - {mode}",
        )

        mlflow.log_metric("test_loss", test_loss)
        for metric_name in ("macro_roc_auc", "micro_roc_auc", "macro_f1", "micro_f1", "macro_average_precision"):
            mlflow.log_metric(f"test_{metric_name}", test_metrics[metric_name])
        mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(output_dir / "resolved_config.yaml"))
        mlflow.log_artifact(str(output_dir / "test_metrics.json"))
        mlflow.log_artifact(str(output_dir / "test_confusion_counts.json"))
        mlflow.log_artifact(str(output_dir / "training_curves.png"))
        mlflow.log_artifact(str(output_dir / "per_class_roc_auc.png"))
        mlflow.log_artifact(str(output_dir / "confusion_matrix.png"))
        if (output_dir / "robustness_metrics.json").exists():
            mlflow.log_artifact(str(output_dir / "robustness_metrics.json"))

    return outputs


def _train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    mode: str,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in tqdm(loader, desc=f"MM Train {epoch}", leave=False):
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(model, batch, device, mode)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_names: list[str],
    mode: str,
    disable_image: bool = False,
    disable_text: bool = False,
    return_outputs: bool = False,
) -> tuple[float, dict[str, Any], dict[str, torch.Tensor] | None]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_probs = []
    all_labels = []

    for batch in tqdm(loader, desc="MM Eval", leave=False):
        labels = batch["labels"].to(device)
        logits = _forward(model, batch, device, mode, disable_image=disable_image, disable_text=disable_text)
        loss = criterion(logits, labels)
        probabilities = torch.sigmoid(logits)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        all_probs.append(probabilities.cpu())
        all_labels.append(labels.cpu())

    labels = torch.cat(all_labels)
    probabilities = torch.cat(all_probs)
    metrics = compute_multilabel_metrics(labels, probabilities, label_names=label_names)
    outputs = None
    if return_outputs:
        outputs = {"labels": labels, "probabilities": probabilities}
    return total_loss / max(total_items, 1), metrics, outputs


def _forward(
    model: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    mode: str,
    disable_image: bool = False,
    disable_text: bool = False,
) -> torch.Tensor:
    image = batch["image"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    if mode in {"image", "image_only"}:
        return model(image)
    if mode in {"text", "text_only"}:
        return model(input_ids, attention_mask)
    if mode in {"fusion", "multimodal"}:
        return model(
            image,
            input_ids,
            attention_mask,
            disable_image=disable_image,
            disable_text=disable_text,
        )
    raise ValueError(f"Unsupported multimodal mode: {mode}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_name = training_cfg.get("scheduler", "none").lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg["epochs"])
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")
