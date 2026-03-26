from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medmnist import ChestMNIST  # noqa: E402
from radiology_triage.data.chestmnist import CHESTMNIST_LABELS, build_transforms  # noqa: E402
from radiology_triage.utils.io import ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA figures for ChestMNIST.")
    parser.add_argument("--root", type=Path, default=Path("data/medmnist"), help="Dataset storage path.")
    parser.add_argument("--size", type=int, default=128, help="ChestMNIST native image size to download.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eda"), help="Output folder for figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    args.root.mkdir(parents=True, exist_ok=True)
    dataset = ChestMNIST(
        split="train",
        root=str(args.root),
        size=args.size,
        as_rgb=True,
        download=True,
        transform=build_transforms(image_size=128, augment=False, normalization="none"),
    )
    labels = torch.as_tensor(dataset.labels, dtype=torch.float32).numpy()
    positives = labels.sum(axis=0)
    frequencies = positives / max(len(labels), 1)

    plt.figure(figsize=(12, 5))
    plt.bar(CHESTMNIST_LABELS, frequencies, color="#2a9d8f")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Positive ratio")
    plt.title("ChestMNIST label distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=200)
    plt.close()

    cooccurrence = np.matmul(labels.T, labels) / max(len(labels), 1)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cooccurrence, xticklabels=CHESTMNIST_LABELS, yticklabels=CHESTMNIST_LABELS, cmap="mako")
    plt.title("Label co-occurrence")
    plt.tight_layout()
    plt.savefig(output_dir / "label_cooccurrence.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for axis, idx in zip(axes.flatten(), range(12)):
        image, label = dataset[idx]
        axis.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        positives = [CHESTMNIST_LABELS[i] for i, value in enumerate(label.flatten()) if value == 1]
        axis.set_title(", ".join(positives[:2]) if positives else "No positive label")
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "sample_images.png", dpi=200)
    plt.close(fig)

    print(f"EDA figures saved to {output_dir}")


if __name__ == "__main__":
    main()
