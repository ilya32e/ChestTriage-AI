from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from radiology_triage.utils.io import ensure_dir, save_json  # noqa: E402


LABEL_MAPPING = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Effusion": "effusion",
    "Infiltration": "infiltration",
    "Mass": "mass",
    "Nodule": "nodule",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Emphysema": "emphysema",
    "Fibrosis": "fibrosis",
    "Pleural_Thickening": "pleural",
    "Hernia": "hernia",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NIH Chest X-rays metadata for the multimodal pipeline.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/nih"),
        help="Destination folder for the generated CSV and summary.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation ratio applied on official train_val patients.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the patient-level train/val split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    dataset_root = Path(kagglehub.dataset_download("nih-chest-xrays/data"))
    data_entry = pd.read_csv(dataset_root / "Data_Entry_2017.csv").rename(
        columns={
            "Image Index": "image_index",
            "Finding Labels": "finding_labels",
            "Follow-up #": "follow_up",
            "Patient ID": "patient_id",
            "Patient Age": "patient_age",
            "Patient Gender": "patient_gender",
            "View Position": "view_position",
        }
    )
    data_entry = data_entry[["image_index", "finding_labels", "follow_up", "patient_id", "patient_age", "patient_gender", "view_position"]].copy()

    bbox = pd.read_csv(dataset_root / "BBox_List_2017.csv").rename(
        columns={
            "Image Index": "image_index",
            "Finding Label": "bbox_label",
            "Bbox [x": "bbox_x",
            "y": "bbox_y",
            "w": "bbox_w",
            "h]": "bbox_h",
        }
    )
    bbox = bbox[["image_index", "bbox_label", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]].copy()
    bbox["bbox_area"] = bbox["bbox_w"] * bbox["bbox_h"]
    bbox_summary = (
        bbox.groupby("image_index")
        .agg(
            bbox_count=("bbox_label", "size"),
            bbox_area_mean=("bbox_area", "mean"),
            bbox_area_max=("bbox_area", "max"),
        )
        .reset_index()
    )

    image_path_map = build_image_path_map(dataset_root)
    data_entry["image_path"] = data_entry["image_index"].map(image_path_map)
    missing_paths = int(data_entry["image_path"].isna().sum())
    if missing_paths:
        raise RuntimeError(f"{missing_paths} NIH images could not be mapped to physical files.")

    dataframe = data_entry.merge(bbox_summary, on="image_index", how="left")
    dataframe["bbox_count"] = dataframe["bbox_count"].fillna(0).astype(int)
    dataframe["bbox_area_mean"] = dataframe["bbox_area_mean"].fillna(0.0)
    dataframe["bbox_area_max"] = dataframe["bbox_area_max"].fillna(0.0)
    dataframe["has_bbox_annotation"] = (dataframe["bbox_count"] > 0).astype(int)

    for source_label, target_label in LABEL_MAPPING.items():
        dataframe[target_label] = dataframe["finding_labels"].apply(
            lambda value, source_label=source_label: int(label_present(value, source_label))
        )

    train_val_images = read_list(dataset_root / "train_val_list.txt")
    test_images = read_list(dataset_root / "test_list.txt")
    dataframe["split"] = dataframe["image_index"].apply(lambda image: "test" if image in test_images else "train_val")

    train_val_df = dataframe[dataframe["split"] == "train_val"].copy()
    patient_split = make_patient_split(train_val_df["patient_id"].unique(), val_ratio=args.val_ratio, seed=args.seed)
    dataframe.loc[dataframe["split"] == "train_val", "split"] = dataframe.loc[dataframe["split"] == "train_val", "patient_id"].map(patient_split)

    dataframe["metadata_text"] = dataframe.apply(build_metadata_text, axis=1)

    ordered_columns = [
        "split",
        "image_path",
        "metadata_text",
        "image_index",
        "patient_id",
        "follow_up",
        "patient_age",
        "patient_gender",
        "view_position",
        "has_bbox_annotation",
        "bbox_count",
        "bbox_area_mean",
        "bbox_area_max",
    ] + list(LABEL_MAPPING.values())
    dataframe = dataframe[ordered_columns].copy()

    csv_path = output_dir / "nih_multimodal_metadata.csv"
    dataframe.to_csv(csv_path, index=False)

    summary = {
        "dataset_root": str(dataset_root),
        "rows": int(len(dataframe)),
        "splits": dataframe["split"].value_counts().to_dict(),
        "patients": int(dataframe["patient_id"].nunique()),
        "missing_image_paths": missing_paths,
        "label_counts": {label: int(dataframe[label].sum()) for label in LABEL_MAPPING.values()},
        "csv_path": str(csv_path),
    }
    save_json(summary, output_dir / "import_summary.json")
    print(json.dumps(summary, indent=2))


def build_image_path_map(dataset_root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for folder in sorted(dataset_root.glob("images_*")):
        image_dir = folder / "images"
        for image_path in image_dir.glob("*.png"):
            mapping[image_path.name] = str(image_path.resolve())
    return mapping


def read_list(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def make_patient_split(patient_ids: np.ndarray, val_ratio: float, seed: int) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    patient_ids = np.array(sorted(patient_ids))
    rng.shuffle(patient_ids)
    val_size = int(len(patient_ids) * val_ratio)
    val_patients = set(patient_ids[:val_size].tolist())
    return {int(patient_id): ("val" if int(patient_id) in val_patients else "train") for patient_id in patient_ids}


def label_present(finding_labels: str, source_label: str) -> bool:
    if pd.isna(finding_labels):
        return False
    labels = {label.strip() for label in str(finding_labels).split("|")}
    return source_label in labels


def build_metadata_text(row: pd.Series) -> str:
    age_years = int(row["patient_age"])
    return (
        f"patient_age {age_years} "
        f"patient_gender {row['patient_gender']} "
        f"view_position {row['view_position']} "
        f"follow_up {int(row['follow_up'])} "
        f"has_bbox_annotation {int(row['has_bbox_annotation'])} "
        f"bbox_count {int(row['bbox_count'])} "
        f"bbox_area_mean {row['bbox_area_mean']:.2f} "
        f"bbox_area_max {row['bbox_area_max']:.2f}"
    )


if __name__ == "__main__":
    main()
