from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from medmnist import ChestMNIST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ChestMNIST with the official MedMNIST API.")
    parser.add_argument("--root", type=Path, default=Path("data/medmnist"), help="Local MedMNIST root directory.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[64, 128, 224],
        help="ChestMNIST sizes to download with the official API.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Dataset splits to prepare.",
    )
    parser.add_argument(
        "--as-rgb",
        action="store_true",
        help="Download RGB samples instead of grayscale.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("data/medmnist/download_summary.json"),
        help="Where to write the download summary JSON.",
    )
    return parser.parse_args()


def build_summary(root: Path, sizes: list[int], splits: list[str], as_rgb: bool) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "dataset": "ChestMNIST",
        "root": str(root.resolve()),
        "sizes": {},
        "as_rgb": bool(as_rgb),
    }

    for size in sizes:
        size_key = str(size)
        summary["sizes"][size_key] = {}
        for split in splits:
            dataset = ChestMNIST(
                split=split,
                root=str(root),
                download=True,
                size=size,
                as_rgb=as_rgb,
            )
            summary["sizes"][size_key][split] = {
                "num_samples": len(dataset),
                "labels_shape": list(dataset.labels.shape),
            }
    return summary


def main() -> None:
    args = parse_args()
    summary = build_summary(
        root=args.root,
        sizes=args.sizes,
        splits=args.splits,
        as_rgb=args.as_rgb,
    )
    summary_path = args.summary_path.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
