from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def to_numpy(data: Any) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _safe_binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _safe_binary_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _safe_binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


def normalize_multilabel_thresholds(
    num_classes: int,
    threshold: float = 0.5,
    thresholds: list[float] | np.ndarray | torch.Tensor | None = None,
) -> np.ndarray:
    if thresholds is None:
        return np.full(num_classes, float(threshold), dtype=np.float32)
    values = to_numpy(thresholds).astype(np.float32).reshape(-1)
    if values.shape[0] != num_classes:
        raise ValueError(f"Expected {num_classes} thresholds, got {values.shape[0]}.")
    return values


def apply_multilabel_thresholds(
    y_prob: np.ndarray | torch.Tensor,
    threshold: float = 0.5,
    thresholds: list[float] | np.ndarray | torch.Tensor | None = None,
) -> np.ndarray:
    probabilities = to_numpy(y_prob).astype(np.float32)
    per_class_thresholds = normalize_multilabel_thresholds(
        probabilities.shape[1],
        threshold=threshold,
        thresholds=thresholds,
    )
    return (probabilities >= per_class_thresholds.reshape(1, -1)).astype(int)


def compute_multilabel_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_prob: np.ndarray | torch.Tensor,
    label_names: list[str] | None = None,
    threshold: float = 0.5,
    thresholds: list[float] | np.ndarray | torch.Tensor | None = None,
) -> dict[str, Any]:
    y_true = to_numpy(y_true).astype(int)
    y_prob = to_numpy(y_prob).astype(np.float32)
    per_class_thresholds = normalize_multilabel_thresholds(
        y_true.shape[1],
        threshold=threshold,
        thresholds=thresholds,
    )
    y_pred = (y_prob >= per_class_thresholds.reshape(1, -1)).astype(int)

    n_classes = y_true.shape[1]
    if label_names is None:
        label_names = [f"class_{idx}" for idx in range(n_classes)]

    per_class: dict[str, dict[str, float | int]] = {}
    roc_scores = []
    ap_scores = []
    for idx, label_name in enumerate(label_names):
        class_true = y_true[:, idx]
        class_prob = y_prob[:, idx]
        class_pred = y_pred[:, idx]
        class_auc = _safe_binary_roc_auc(class_true, class_prob)
        class_ap = _safe_binary_average_precision(class_true, class_prob)
        roc_scores.append(class_auc)
        ap_scores.append(class_ap)
        per_class[label_name] = {
            "roc_auc": class_auc,
            "average_precision": class_ap,
            "f1": _safe_binary_f1(class_true, class_pred),
            "support": int(class_true.sum()),
            "threshold": float(per_class_thresholds[idx]),
        }

    flat_true = y_true.reshape(-1)
    flat_prob = y_prob.reshape(-1)
    flat_pred = y_pred.reshape(-1)

    macro_roc_auc = float(np.nanmean(roc_scores)) if roc_scores else float("nan")
    macro_average_precision = float(np.nanmean(ap_scores)) if ap_scores else float("nan")
    micro_roc_auc = _safe_binary_roc_auc(flat_true, flat_prob)
    micro_average_precision = _safe_binary_average_precision(flat_true, flat_prob)

    return {
        "macro_roc_auc": macro_roc_auc,
        "micro_roc_auc": micro_roc_auc,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_average_precision": macro_average_precision,
        "micro_average_precision": micro_average_precision,
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold_mode": "per_class" if thresholds is not None else "global",
        "thresholds": {label_name: float(per_class_thresholds[idx]) for idx, label_name in enumerate(label_names)},
        "per_class": per_class,
    }


def compute_multilabel_confusion_counts(
    y_true: np.ndarray | torch.Tensor,
    y_prob: np.ndarray | torch.Tensor,
    label_names: list[str] | None = None,
    threshold: float = 0.5,
    thresholds: list[float] | np.ndarray | torch.Tensor | None = None,
) -> dict[str, dict[str, int]]:
    y_true = to_numpy(y_true).astype(int)
    y_prob = to_numpy(y_prob).astype(np.float32)
    per_class_thresholds = normalize_multilabel_thresholds(
        y_true.shape[1],
        threshold=threshold,
        thresholds=thresholds,
    )
    y_pred = (y_prob >= per_class_thresholds.reshape(1, -1)).astype(int)

    n_classes = y_true.shape[1]
    if label_names is None:
        label_names = [f"class_{idx}" for idx in range(n_classes)]

    confusion: dict[str, dict[str, int]] = {}
    for idx, label_name in enumerate(label_names):
        class_true = y_true[:, idx]
        class_pred = y_pred[:, idx]
        confusion[label_name] = {
            "tn": int(np.sum((class_true == 0) & (class_pred == 0))),
            "fp": int(np.sum((class_true == 0) & (class_pred == 1))),
            "fn": int(np.sum((class_true == 1) & (class_pred == 0))),
            "tp": int(np.sum((class_true == 1) & (class_pred == 1))),
            "threshold": float(per_class_thresholds[idx]),
        }
    return confusion


def compute_anomaly_metrics(
    y_true: np.ndarray | torch.Tensor,
    scores: np.ndarray | torch.Tensor,
    threshold: float | None = None,
) -> dict[str, float]:
    y_true = to_numpy(y_true).astype(int)
    scores = to_numpy(scores).astype(np.float32)
    metrics = {
        "roc_auc": _safe_binary_roc_auc(y_true, scores),
        "average_precision": _safe_binary_average_precision(y_true, scores),
    }
    if threshold is not None:
        y_pred = (scores >= threshold).astype(int)
        metrics.update(
            {
                "threshold": float(threshold),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
            }
        )
    return metrics


def compute_binary_confusion_counts(
    y_true: np.ndarray | torch.Tensor,
    scores: np.ndarray | torch.Tensor,
    threshold: float,
) -> dict[str, int]:
    y_true = to_numpy(y_true).astype(int)
    scores = to_numpy(scores).astype(np.float32)
    y_pred = (scores >= threshold).astype(int)
    return {
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
    }


def calibrate_anomaly_threshold(
    normal_scores: np.ndarray | torch.Tensor,
    quantile: float = 0.95,
) -> float:
    normal_scores = to_numpy(normal_scores).astype(np.float32)
    return float(np.quantile(normal_scores, quantile))


def calibrate_multilabel_thresholds(
    y_true: np.ndarray | torch.Tensor,
    y_prob: np.ndarray | torch.Tensor,
    label_names: list[str] | None = None,
    default_threshold: float = 0.5,
    search_space: list[float] | np.ndarray | torch.Tensor | None = None,
) -> dict[str, Any]:
    y_true = to_numpy(y_true).astype(int)
    y_prob = to_numpy(y_prob).astype(np.float32)

    n_classes = y_true.shape[1]
    if label_names is None:
        label_names = [f"class_{idx}" for idx in range(n_classes)]

    if search_space is None:
        search_values = np.linspace(0.05, 0.95, 91, dtype=np.float32)
    else:
        search_values = np.unique(np.clip(to_numpy(search_space).astype(np.float32).reshape(-1), 0.0, 1.0))
    if default_threshold not in search_values:
        search_values = np.unique(np.append(search_values, np.float32(default_threshold)))

    thresholds = np.full(n_classes, float(default_threshold), dtype=np.float32)
    per_class: dict[str, dict[str, float | int]] = {}

    for idx, label_name in enumerate(label_names):
        class_true = y_true[:, idx]
        class_prob = y_prob[:, idx]
        best_threshold = float(default_threshold)
        best_f1 = float("-inf")
        best_precision = 0.0
        best_recall = 0.0

        for candidate in search_values:
            class_pred = (class_prob >= candidate).astype(int)
            current_f1 = float(f1_score(class_true, class_pred, zero_division=0))
            current_precision = float(precision_score(class_true, class_pred, zero_division=0))
            current_recall = float(recall_score(class_true, class_pred, zero_division=0))
            if current_f1 > best_f1:
                best_threshold = float(candidate)
                best_f1 = current_f1
                best_precision = current_precision
                best_recall = current_recall
                continue
            if np.isclose(current_f1, best_f1) and abs(float(candidate) - default_threshold) < abs(best_threshold - default_threshold):
                best_threshold = float(candidate)
                best_precision = current_precision
                best_recall = current_recall

        thresholds[idx] = best_threshold
        per_class[label_name] = {
            "threshold": best_threshold,
            "f1": float(best_f1),
            "precision": best_precision,
            "recall": best_recall,
            "support": int(class_true.sum()),
        }

    default_metrics = compute_multilabel_metrics(
        y_true,
        y_prob,
        label_names=label_names,
        threshold=default_threshold,
    )
    calibrated_metrics = compute_multilabel_metrics(
        y_true,
        y_prob,
        label_names=label_names,
        thresholds=thresholds,
    )

    return {
        "strategy": "per_class_f1_grid_search",
        "default_threshold": float(default_threshold),
        "search_space": [float(value) for value in search_values.tolist()],
        "thresholds": {label_name: float(thresholds[idx]) for idx, label_name in enumerate(label_names)},
        "ordered_thresholds": [float(value) for value in thresholds.tolist()],
        "validation_metrics_default": default_metrics,
        "validation_metrics_calibrated": calibrated_metrics,
        "macro_f1_gain": float(calibrated_metrics["macro_f1"] - default_metrics["macro_f1"]),
        "per_class": per_class,
    }


def top_k_predictions(probabilities: np.ndarray | torch.Tensor, label_names: list[str], k: int = 5) -> list[tuple[str, float]]:
    probabilities = to_numpy(probabilities).astype(np.float32)
    order = np.argsort(probabilities)[::-1][:k]
    return [(label_names[idx], float(probabilities[idx])) for idx in order]
