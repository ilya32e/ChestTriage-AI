from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from radiology_triage.config import flatten_dict, save_yaml
from radiology_triage.data.chestmnist import build_chestmnist_loaders, compute_pos_weight, subset_labels
from radiology_triage.models.supervised import build_supervised_model, count_trainable_parameters, freeze_backbone
from radiology_triage.utils.io import ensure_dir, save_checkpoint, save_json
from radiology_triage.utils.metrics import compute_multilabel_confusion_counts, compute_multilabel_metrics
from radiology_triage.utils.mlflow_utils import setup_mlflow
from radiology_triage.utils.plotting import (
    save_multilabel_confusion_grid,
    save_per_class_metric_plot,
    save_training_curves,
)
from radiology_triage.utils.repro import get_device, seed_everything


def run_supervised_experiment(config: dict[str, Any]) -> dict[str, Any]:
    seed_everything(config.get("seed", 42))
    device = get_device()
    output_dir = ensure_dir(config["output_dir"])
    save_yaml(config, output_dir / "resolved_config.yaml")

    loaders = build_chestmnist_loaders(config)
    model_cfg = config["model"]
    model = build_supervised_model(
        model_name=model_cfg["name"],
        num_labels=len(loaders.label_names),
        pretrained=model_cfg.get("pretrained", False),
        dropout=model_cfg.get("dropout", 0.2),
        in_channels=3 if config["dataset"].get("as_rgb", True) else 1,
        image_size=config["dataset"]["image_size"],
    )
    model = model.to(device)

    training_cfg = config["training"]
    pos_weight = None
    if training_cfg.get("use_pos_weight", True):
        pos_weight = compute_pos_weight(subset_labels(loaders.train.dataset)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = _build_optimizer(model, model_cfg, training_cfg)
    scheduler = _build_scheduler(optimizer, training_cfg)

    setup_mlflow(config["tracking_uri"], config["experiment_name"])
    run_name = config.get("run_name") or f"{model_cfg['name']}_{config['dataset']['size']}"

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_roc_auc": [],
        "val_macro_f1": [],
        "val_macro_average_precision": [],
    }
    best_score = float("-inf")
    best_epoch = 0
    best_checkpoint_path = Path(output_dir) / "best_model.pt"
    patience = training_cfg.get("early_stopping_patience", 5)
    patience_counter = 0

    with mlflow.start_run(run_name=run_name) as active_run:
        run_id = active_run.info.run_id
        mlflow.log_params(flatten_dict(config))
        mlflow.log_param("device", str(device))
        mlflow.log_param("trainable_parameters", count_trainable_parameters(model))

        for epoch in range(1, training_cfg["epochs"] + 1):
            train_loss = _train_one_epoch(model, loaders.train, optimizer, criterion, device, epoch)
            val_loss, val_metrics, _ = _evaluate(model, loaders.val, criterion, device, loaders.label_names)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_macro_roc_auc"].append(val_metrics["macro_roc_auc"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])
            history["val_macro_average_precision"].append(val_metrics["macro_average_precision"])

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

            current_score = val_metrics["macro_roc_auc"]
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(
                    best_checkpoint_path,
                    {
                        "state_dict": model.state_dict(),
                        "config": config,
                        "label_names": loaders.label_names,
                        "model_name": model_cfg["name"],
                        "dropout": model_cfg.get("dropout", 0.2),
                        "image_size": config["dataset"]["image_size"],
                        "mlflow_run_id": run_id,
                        "mlflow_experiment_name": config["experiment_name"],
                        "mlflow_run_name": run_name,
                    },
                )
                mlflow.log_metric("best_val_macro_roc_auc", best_score, step=epoch)
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
            return_outputs=True,
        )
        if test_outputs is None:
            raise RuntimeError("Expected test outputs for confusion matrix generation.")
        test_confusion = compute_multilabel_confusion_counts(
            test_outputs["labels"],
            test_outputs["probabilities"],
            label_names=loaders.label_names,
        )

        save_json(test_metrics, output_dir / "test_metrics.json")
        save_json(test_confusion, output_dir / "test_confusion_counts.json")
        save_training_curves(history, output_dir / "training_curves.png")
        save_per_class_metric_plot(test_metrics["per_class"], "roc_auc", output_dir / "per_class_roc_auc.png")
        save_per_class_metric_plot(
            test_metrics["per_class"],
            "average_precision",
            output_dir / "per_class_average_precision.png",
        )
        save_multilabel_confusion_grid(
            test_confusion,
            output_dir / "confusion_matrix.png",
            title=f"Confusion matrices - {model_cfg['name']}",
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
        mlflow.log_artifact(str(output_dir / "per_class_average_precision.png"))
        mlflow.log_artifact(str(output_dir / "confusion_matrix.png"))

    return {
        "best_epoch": best_epoch,
        "best_val_macro_roc_auc": best_score,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "checkpoint_path": str(best_checkpoint_path),
        "mlflow_run_id": run_id,
    }


def _train_one_epoch(
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
    for images, labels in tqdm(loader, desc=f"Train {epoch}", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
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
    return_outputs: bool = False,
) -> tuple[float, dict[str, Any], dict[str, torch.Tensor] | None]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        probabilities = torch.sigmoid(logits)

        batch_size = images.size(0)
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


def _build_optimizer(
    model: nn.Module,
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    model_name = model_cfg["name"].lower()
    fine_tuning_mode = model_cfg.get("fine_tuning_mode", "none").lower()
    weight_decay = training_cfg.get("weight_decay", 0.0)

    if fine_tuning_mode in {"", "none"}:
        if model_cfg.get("freeze_backbone", False):
            freeze_backbone(model, model_name)
        return torch.optim.AdamW(
            filter(lambda parameter: parameter.requires_grad, model.parameters()),
            lr=training_cfg["lr"],
            weight_decay=weight_decay,
        )

    if fine_tuning_mode == "resnet18_layer4_fc":
        if model_name != "resnet18":
            raise ValueError("The resnet18_layer4_fc fine-tuning mode only supports ResNet18.")
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

        head_lr = training_cfg.get("head_lr", training_cfg["lr"])
        backbone_lr = training_cfg.get("backbone_lr", head_lr * 0.1)
        parameter_groups = [
            {
                "params": [parameter for parameter in model.layer4.parameters() if parameter.requires_grad],
                "lr": backbone_lr,
            },
            {
                "params": [parameter for parameter in model.fc.parameters() if parameter.requires_grad],
                "lr": head_lr,
            },
        ]
        return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)

    raise ValueError(f"Unsupported fine-tuning mode: {fine_tuning_mode}")
