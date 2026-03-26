from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from radiology_triage.config import flatten_dict, save_yaml
from radiology_triage.data.chestmnist import build_anomaly_loaders
from radiology_triage.models.autoencoder import build_reconstruction_model
from radiology_triage.utils.io import ensure_dir, save_checkpoint, save_json
from radiology_triage.utils.metrics import (
    calibrate_anomaly_threshold,
    compute_anomaly_metrics,
    compute_binary_confusion_counts,
)
from radiology_triage.utils.mlflow_utils import setup_mlflow
from radiology_triage.utils.plotting import (
    save_anomaly_score_histogram,
    save_binary_confusion_matrix,
    save_reconstruction_grid,
    save_training_curves,
)
from radiology_triage.utils.repro import get_device, seed_everything


def run_anomaly_experiment(config: dict[str, Any]) -> dict[str, Any]:
    seed_everything(config.get("seed", 42))
    device = get_device()
    output_dir = ensure_dir(config["output_dir"])
    save_yaml(config, output_dir / "resolved_config.yaml")

    loaders = build_anomaly_loaders(config)
    model = build_reconstruction_model(
        model_name=config["model"].get("name", "conv_autoencoder"),
        in_channels=1 if not config["dataset"].get("as_rgb", False) else 3,
        latent_channels=config["model"].get("latent_channels", 128),
    ).to(device)

    training_cfg = config["training"]
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["lr"],
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_roc_auc": [],
        "val_average_precision": [],
    }
    best_score = float("-inf")
    best_epoch = 0
    best_threshold = 0.0
    best_checkpoint_path = Path(output_dir) / "best_autoencoder.pt"
    patience = training_cfg.get("early_stopping_patience", 5)
    patience_counter = 0

    setup_mlflow(config["tracking_uri"], config["experiment_name"])

    run_name = config.get("run_name") or "conv_autoencoder"
    with mlflow.start_run(run_name=run_name) as active_run:
        run_id = active_run.info.run_id
        mlflow.log_params(flatten_dict(config))
        mlflow.log_param("device", str(device))

        for epoch in range(1, training_cfg["epochs"] + 1):
            train_loss = _train_autoencoder_epoch(model, loaders.train, optimizer, criterion, device, epoch)
            val_scores, val_targets, val_examples = _score_anomalies(model, loaders.val, device)
            normal_scores = val_scores[val_targets == 0]
            threshold = calibrate_anomaly_threshold(normal_scores, config["evaluation"].get("threshold_quantile", 0.95))
            val_metrics = compute_anomaly_metrics(val_targets, val_scores, threshold=threshold)

            history["train_loss"].append(train_loss)
            history["val_roc_auc"].append(val_metrics["roc_auc"])
            history["val_average_precision"].append(val_metrics["average_precision"])
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_average_precision": val_metrics["average_precision"],
                    "val_f1": val_metrics["f1"],
                },
                step=epoch,
            )

            if val_metrics["roc_auc"] > best_score:
                best_score = val_metrics["roc_auc"]
                best_epoch = epoch
                best_threshold = threshold
                patience_counter = 0
                save_checkpoint(
                    best_checkpoint_path,
                    {
                        "state_dict": model.state_dict(),
                        "config": config,
                        "threshold": best_threshold,
                        "image_size": config["dataset"]["image_size"],
                        "in_channels": 1 if not config["dataset"].get("as_rgb", False) else 3,
                        "mlflow_run_id": run_id,
                        "mlflow_experiment_name": config["experiment_name"],
                        "mlflow_run_name": run_name,
                    },
                )
                save_reconstruction_grid(
                    val_examples["originals"],
                    val_examples["reconstructions"],
                    output_dir / "reconstructions.png",
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        test_scores, test_targets, _ = _score_anomalies(model, loaders.test, device)
        test_metrics = compute_anomaly_metrics(test_targets, test_scores, threshold=best_threshold)
        test_confusion = compute_binary_confusion_counts(test_targets, test_scores, threshold=best_threshold)

        save_json(test_metrics, output_dir / "test_metrics.json")
        save_json(test_confusion, output_dir / "test_confusion_counts.json")
        save_training_curves(history, output_dir / "training_curves.png")
        save_anomaly_score_histogram(
            normal_scores=test_scores[test_targets == 0],
            abnormal_scores=test_scores[test_targets == 1],
            path=output_dir / "anomaly_scores.png",
            threshold=best_threshold,
        )
        save_binary_confusion_matrix(
            test_confusion,
            output_dir / "confusion_matrix.png",
            title="Anomaly confusion matrix",
        )

        mlflow.log_metric("best_val_roc_auc", best_score)
        mlflow.log_metric("test_roc_auc", test_metrics["roc_auc"])
        mlflow.log_metric("test_average_precision", test_metrics["average_precision"])
        mlflow.log_metric("test_f1", test_metrics["f1"])
        mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(output_dir / "resolved_config.yaml"))
        mlflow.log_artifact(str(output_dir / "test_metrics.json"))
        mlflow.log_artifact(str(output_dir / "test_confusion_counts.json"))
        mlflow.log_artifact(str(output_dir / "training_curves.png"))
        if (output_dir / "reconstructions.png").exists():
            mlflow.log_artifact(str(output_dir / "reconstructions.png"))
        mlflow.log_artifact(str(output_dir / "anomaly_scores.png"))
        mlflow.log_artifact(str(output_dir / "confusion_matrix.png"))

    return {
        "best_epoch": best_epoch,
        "best_val_roc_auc": best_score,
        "best_threshold": best_threshold,
        "test_metrics": test_metrics,
        "checkpoint_path": str(best_checkpoint_path),
        "mlflow_run_id": run_id,
    }


def _train_autoencoder_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, _ in tqdm(loader, desc=f"AE Train {epoch}", leave=False):
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        reconstructions = model(images)
        loss = criterion(reconstructions, images)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


@torch.no_grad()
def _score_anomalies(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, torch.Tensor]]:
    model.eval()
    scores = []
    targets = []
    examples: dict[str, torch.Tensor] | None = None

    for images, anomaly_labels, _ in tqdm(loader, desc="AE Eval", leave=False):
        images = images.to(device)
        reconstructions = model(images)
        errors = torch.mean(torch.abs(reconstructions - images), dim=(1, 2, 3))
        scores.append(errors.cpu())
        targets.append(anomaly_labels.cpu())
        if examples is None:
            examples = {
                "originals": images[:6].cpu(),
                "reconstructions": reconstructions[:6].cpu(),
            }

    if examples is None:
        raise RuntimeError("No validation examples were collected.")
    return torch.cat(scores).numpy(), torch.cat(targets).numpy(), examples
