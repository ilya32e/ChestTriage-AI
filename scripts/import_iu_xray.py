from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from radiology_triage.utils.io import ensure_dir, save_json  # noqa: E402


REPO_ID = "dz-osamu/IU-Xray"
SPLIT_FILES = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}
LABEL_PATTERNS = {
    "atelectasis": [r"atelecta(?:sis|tic)"],
    "cardiomegaly": [
        r"cardiomegaly",
        r"cardiac silhouette (?:is )?(?:mildly )?enlarged",
        r"heart size (?:is )?(?:mildly )?enlarged",
    ],
    "effusion": [r"pleural effusions?", r"\beffusions?\b"],
    "edema": [r"pulmonary edema", r"\bedema\b"],
    "pneumothorax": [r"pneumothorax"],
    "hernia": [r"hiatal hernia", r"\bhernia\b"],
}
NEGATION_CUES = ("no", "without", "free of", "negative for", "absence of", "not")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import the IU X-Ray multimodal dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/iu_xray"),
        help="Destination folder for images and generated CSV.",
    )
    parser.add_argument(
        "--duplicate-images",
        action="store_true",
        help="Create one row per image instead of keeping only the first image of each report.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract the ZIP even if images already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    images_dir = ensure_dir(output_dir / "images")

    jsonl_paths = {
        split: Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename))
        for split, filename in SPLIT_FILES.items()
    }
    zip_path = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename="image.zip"))

    if args.force_extract or not any(images_dir.rglob("*.png")):
        extract_images(zip_path, images_dir)

    rows = []
    missing_images = 0
    for split, jsonl_path in jsonl_paths.items():
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in tqdm(handle, desc=f"Parse {split}", leave=False):
                record = json.loads(line)
                report_text = clean_report(record["response"])
                image_refs = record["images"]
                if not image_refs:
                    continue

                labels = derive_labels(report_text)
                selected_refs = image_refs if args.duplicate_images else image_refs[:1]
                for image_ref in selected_refs:
                    relative_image_path = normalize_image_path(image_ref)
                    absolute_image_path = output_dir / relative_image_path
                    if not absolute_image_path.exists():
                        missing_images += 1
                        continue
                    row = {
                        "split": split,
                        "image_path": relative_image_path.as_posix(),
                        "report_text": report_text,
                    }
                    row.update(labels)
                    rows.append(row)

    dataframe = pd.DataFrame(rows)
    csv_path = output_dir / "iu_xray_multimodal.csv"
    dataframe.to_csv(csv_path, index=False)

    label_counts = {label: int(dataframe[label].sum()) for label in LABEL_PATTERNS}
    summary = {
        "rows": int(len(dataframe)),
        "splits": dataframe["split"].value_counts().to_dict(),
        "label_counts": label_counts,
        "missing_images": missing_images,
        "csv_path": str(csv_path),
    }
    save_json(summary, output_dir / "import_summary.json")

    print(json.dumps(summary, indent=2))


def extract_images(zip_path: Path, images_dir: Path) -> None:
    with ZipFile(zip_path) as archive:
        image_members = [member for member in archive.namelist() if member.lower().endswith(".png")]
        for member in tqdm(image_members, desc="Extract images"):
            relative_path = normalize_image_path(member)
            destination = images_dir.parent / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                continue
            with archive.open(member) as source, destination.open("wb") as target:
                target.write(source.read())


def normalize_image_path(raw_path: str) -> Path:
    path = PurePosixPath(raw_path)
    if len(path.parts) < 2:
        raise ValueError(f"Unexpected image path: {raw_path}")
    study_id = path.parts[-2]
    filename = path.parts[-1]
    return Path("images") / study_id / filename


def clean_report(report_text: str) -> str:
    return " ".join((report_text or "").strip().split())


def derive_labels(report_text: str) -> dict[str, int]:
    lowered = report_text.lower()
    return {label: int(has_positive_mention(lowered, patterns)) for label, patterns in LABEL_PATTERNS.items()}


def has_positive_mention(text: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            if is_negated(text, match.start()):
                continue
            return True
    return False


def is_negated(text: str, mention_start: int) -> bool:
    context = text[max(0, mention_start - 80) : mention_start]
    negation_pattern = r"(?:" + "|".join(re.escape(cue) for cue in NEGATION_CUES) + r")\s+(?:\w+\s+){0,6}$"
    return bool(re.search(negation_pattern, context))


if __name__ == "__main__":
    main()
