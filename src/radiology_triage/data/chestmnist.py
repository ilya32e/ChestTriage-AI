from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from medmnist import INFO, ChestMNIST
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


CHESTMNIST_LABELS = [INFO["chestmnist"]["label"][str(idx)] for idx in range(14)]


class ChestMNISTWrapper(Dataset):
    def __init__(self, dataset: ChestMNIST):
        self.dataset = dataset
        self.labels = torch.as_tensor(dataset.labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, labels = self.dataset[index]
        return image, torch.as_tensor(labels, dtype=torch.float32).flatten()


class BinaryAnomalyWrapper(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, labels = self.dataset[index]
        anomaly_label = torch.tensor(float(labels.sum() > 0), dtype=torch.float32)
        return image, anomaly_label, labels


@dataclass
class ChestMNISTLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    label_names: list[str]


@dataclass
class AnomalyLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    label_names: list[str]


def build_transforms(
    image_size: int,
    augment: bool,
    normalization: str = "imagenet",
) -> transforms.Compose:
    mean, std = _resolve_normalization(normalization)
    steps: list[transforms.Compose | transforms.Resize | transforms.RandomHorizontalFlip | transforms.RandomRotation | transforms.ToTensor | transforms.Normalize] = [
        transforms.Resize((image_size, image_size), antialias=True),
    ]
    if augment:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=8),
            ]
        )
    steps.append(transforms.ToTensor())
    if mean is not None and std is not None:
        steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(steps)


def build_chestmnist_loaders(config: dict) -> ChestMNISTLoaders:
    dataset_cfg = config["dataset"]
    root_dir = Path(dataset_cfg["root"])
    root_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = ChestMNISTWrapper(
        ChestMNIST(
            split="train",
            root=str(root_dir),
            size=dataset_cfg["size"],
            as_rgb=dataset_cfg.get("as_rgb", True),
            download=dataset_cfg.get("download", True),
            transform=build_transforms(
                image_size=dataset_cfg["image_size"],
                augment=True,
                normalization=dataset_cfg.get("normalization", "imagenet"),
            ),
        )
    )
    val_dataset = ChestMNISTWrapper(
        ChestMNIST(
            split="val",
            root=str(root_dir),
            size=dataset_cfg["size"],
            as_rgb=dataset_cfg.get("as_rgb", True),
            download=dataset_cfg.get("download", True),
            transform=build_transforms(
                image_size=dataset_cfg["image_size"],
                augment=False,
                normalization=dataset_cfg.get("normalization", "imagenet"),
            ),
        )
    )
    test_dataset = ChestMNISTWrapper(
        ChestMNIST(
            split="test",
            root=str(root_dir),
            size=dataset_cfg["size"],
            as_rgb=dataset_cfg.get("as_rgb", True),
            download=dataset_cfg.get("download", True),
            transform=build_transforms(
                image_size=dataset_cfg["image_size"],
                augment=False,
                normalization=dataset_cfg.get("normalization", "imagenet"),
            ),
        )
    )
    train_dataset = _maybe_limit_dataset(train_dataset, dataset_cfg.get("max_train_samples"), config.get("seed", 42))
    val_dataset = _maybe_limit_dataset(val_dataset, dataset_cfg.get("max_val_samples"), config.get("seed", 42) + 1)
    test_dataset = _maybe_limit_dataset(test_dataset, dataset_cfg.get("max_test_samples"), config.get("seed", 42) + 2)

    loader_kwargs = {
        "batch_size": dataset_cfg["batch_size"],
        "num_workers": dataset_cfg.get("num_workers", 0),
        "pin_memory": bool(torch.cuda.is_available()),
    }
    return ChestMNISTLoaders(
        train=DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        val=DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        test=DataLoader(test_dataset, shuffle=False, **loader_kwargs),
        label_names=CHESTMNIST_LABELS,
    )


def build_anomaly_loaders(config: dict) -> AnomalyLoaders:
    dataset_cfg = config["dataset"]
    root_dir = Path(dataset_cfg["root"])
    root_dir.mkdir(parents=True, exist_ok=True)
    train_base = ChestMNISTWrapper(
        ChestMNIST(
            split="train",
            root=str(root_dir),
            size=dataset_cfg["size"],
            as_rgb=dataset_cfg.get("as_rgb", False),
            download=dataset_cfg.get("download", True),
            transform=build_transforms(
                image_size=dataset_cfg["image_size"],
                augment=True,
                normalization=dataset_cfg.get("normalization", "none"),
            ),
        )
    )
    val_base = ChestMNISTWrapper(
        ChestMNIST(
            split="val",
            root=str(root_dir),
            size=dataset_cfg["size"],
            as_rgb=dataset_cfg.get("as_rgb", False),
            download=dataset_cfg.get("download", True),
            transform=build_transforms(
                image_size=dataset_cfg["image_size"],
                augment=False,
                normalization=dataset_cfg.get("normalization", "none"),
            ),
        )
    )
    test_base = ChestMNISTWrapper(
        ChestMNIST(
            split="test",
            root=str(root_dir),
            size=dataset_cfg["size"],
            as_rgb=dataset_cfg.get("as_rgb", False),
            download=dataset_cfg.get("download", True),
            transform=build_transforms(
                image_size=dataset_cfg["image_size"],
                augment=False,
                normalization=dataset_cfg.get("normalization", "none"),
            ),
        )
    )

    normal_indices = torch.where(train_base.labels.sum(dim=1) == 0)[0].tolist()
    if not normal_indices:
        raise RuntimeError("No normal images found in the ChestMNIST training split.")
    train_dataset = Subset(train_base, normal_indices)
    train_dataset = _maybe_limit_dataset(train_dataset, dataset_cfg.get("max_train_samples"), config.get("seed", 42))
    val_dataset = BinaryAnomalyWrapper(
        _maybe_limit_dataset(val_base, dataset_cfg.get("max_val_samples"), config.get("seed", 42) + 1)
    )
    test_dataset = BinaryAnomalyWrapper(
        _maybe_limit_dataset(test_base, dataset_cfg.get("max_test_samples"), config.get("seed", 42) + 2)
    )

    loader_kwargs = {
        "batch_size": dataset_cfg["batch_size"],
        "num_workers": dataset_cfg.get("num_workers", 0),
        "pin_memory": bool(torch.cuda.is_available()),
    }
    return AnomalyLoaders(
        train=DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        val=DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        test=DataLoader(test_dataset, shuffle=False, **loader_kwargs),
        label_names=CHESTMNIST_LABELS,
    )


def compute_pos_weight(labels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    return negatives / (positives + eps)


def subset_labels(dataset: Dataset) -> torch.Tensor:
    if isinstance(dataset, Subset):
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        parent_labels = subset_labels(dataset.dataset)
        return parent_labels[indices]
    if hasattr(dataset, "labels"):
        return dataset.labels
    raise TypeError(f"Unsupported dataset type: {type(dataset)!r}")


def _resolve_normalization(normalization: str) -> tuple[list[float] | None, list[float] | None]:
    mapping: dict[str, tuple[list[float] | None, list[float] | None]] = {
        "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "grayscale": ([0.5], [0.5]),
        "none": (None, None),
    }
    if normalization not in mapping:
        raise ValueError(f"Unknown normalization mode: {normalization}")
    return mapping[normalization]


def _maybe_limit_dataset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None:
        return dataset
    if max_samples <= 0:
        raise ValueError("max_samples must be a positive integer.")
    if len(dataset) <= max_samples:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)
