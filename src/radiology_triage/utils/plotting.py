from __future__ import annotations

from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_training_curves(history: dict[str, list[float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    for key, values in history.items():
        if values:
            plt.plot(range(1, len(values) + 1), values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_per_class_metric_plot(
    per_class: dict[str, dict[str, float | int]],
    metric_name: str,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(per_class.keys())
    values = [per_class[label].get(metric_name, float("nan")) for label in labels]
    plt.figure(figsize=(12, 5))
    plt.bar(labels, values, color="#1f77b4")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(f"Per-class {metric_name.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_anomaly_score_histogram(
    normal_scores: np.ndarray,
    abnormal_scores: np.ndarray,
    path: str | Path,
    threshold: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.hist(normal_scores, bins=30, alpha=0.6, label="Normal")
    plt.hist(abnormal_scores, bins=30, alpha=0.6, label="Abnormal")
    if threshold is not None:
        plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.4f}")
    plt.xlabel("Anomaly score")
    plt.ylabel("Count")
    plt.title("Anomaly score distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_reconstruction_grid(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    path: str | Path,
    max_items: int = 6,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    originals = originals.detach().cpu()[:max_items]
    reconstructions = reconstructions.detach().cpu()[:max_items]

    rows = len(originals)
    plt.figure(figsize=(8, 2.5 * rows))
    for row in range(rows):
        plt.subplot(rows, 2, 2 * row + 1)
        plt.imshow(_tensor_to_image(originals[row]), cmap="gray")
        plt.axis("off")
        plt.title("Original")

        plt.subplot(rows, 2, 2 * row + 2)
        plt.imshow(_tensor_to_image(reconstructions[row]), cmap="gray")
        plt.axis("off")
        plt.title("Reconstruction")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_binary_confusion_matrix(
    confusion: dict[str, int],
    path: str | Path,
    title: str = "Binary confusion matrix",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(
        [
            [confusion["tn"], confusion["fp"]],
            [confusion["fn"], confusion["tp"]],
        ],
        dtype=np.float32,
    )
    display = _normalize_confusion(matrix)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(display, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title(title)
    _annotate_confusion(ax, matrix, display)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_multilabel_confusion_grid(
    per_class_confusion: dict[str, dict[str, int]],
    path: str | Path,
    title: str = "Per-class confusion matrices",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(per_class_confusion.keys())
    if not labels:
        raise ValueError("per_class_confusion cannot be empty")

    cols = min(4, max(1, len(labels)))
    rows = math.ceil(len(labels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for index, label in enumerate(labels):
        row = index // cols
        col = index % cols
        ax = axes[row, col]
        confusion = per_class_confusion[label]
        matrix = np.array(
            [
                [confusion["tn"], confusion["fp"]],
                [confusion["fn"], confusion["tp"]],
            ],
            dtype=np.float32,
        )
        display = _normalize_confusion(matrix)
        ax.imshow(display, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        ax.set_title(label)
        _annotate_confusion(ax, matrix, display)

    for index in range(len(labels), rows * cols):
        row = index // cols
        col = index % cols
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(path, dpi=200)
    plt.close()


def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.numpy()
    if array.shape[0] == 1:
        return np.clip(array[0], 0.0, 1.0)
    array = np.transpose(array, (1, 2, 0))
    return np.clip(array, 0.0, 1.0)


def _normalize_confusion(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def _annotate_confusion(ax: plt.Axes, counts: np.ndarray, normalized: np.ndarray) -> None:
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            value = counts[i, j]
            proportion = normalized[i, j]
            color = "white" if proportion >= 0.5 else "black"
            ax.text(
                j,
                i,
                f"{int(value)}\n{proportion:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
