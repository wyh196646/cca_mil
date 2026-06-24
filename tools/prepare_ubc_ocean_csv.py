#!/usr/bin/env python3
# coding=utf-8
"""Prepare dataset_csv/UBC-OCEAN.csv from UBC-OCEAN metadata."""

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_DIR = Path("/data2/yuhaowang/WSIFew/processd_wsi/UBC-OCEAN/feature/pt_files")
DEFAULT_OUTPUT = ROOT / "dataset_csv" / "UBC-OCEAN.csv"
VALID_LABELS = {"CC", "HGSC", "LGSC", "EC", "MC"}


def strip_slide_suffix(value):
    text = str(value).strip()
    for suffix in (".svs", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".pt"):
        if text.lower().endswith(suffix):
            return text[: -len(suffix)]
    return text


def find_column(fieldnames, candidates):
    lookup = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def load_feature_ids(feature_dir):
    if not feature_dir.is_dir():
        raise FileNotFoundError("UBC feature dir not found: {}".format(feature_dir))
    ids = {path.stem for path in feature_dir.glob("*.pt")}
    if not ids:
        raise FileNotFoundError("No .pt feature files found in {}".format(feature_dir))
    return ids


def build_rows(metadata, feature_dir, include_missing_features=False):
    feature_ids = load_feature_ids(feature_dir)
    with metadata.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError("Metadata CSV has no header: {}".format(metadata))

        slide_col = find_column(fieldnames, ("slide_id", "image_id", "image", "id"))
        label_col = find_column(fieldnames, ("label", "subtype", "diagnosis"))
        case_col = find_column(fieldnames, ("case_id", "patient_id", "image_id", "slide_id", "id"))
        if slide_col is None or label_col is None:
            raise ValueError(
                "Metadata must contain slide/image id and label columns. "
                "Found columns: {}".format(fieldnames)
            )

        rows = []
        missing_features = []
        bad_labels = []
        for row in reader:
            slide_id = strip_slide_suffix(row.get(slide_col, ""))
            if not slide_id:
                continue
            label = str(row.get(label_col, "")).strip()
            if label not in VALID_LABELS:
                bad_labels.append((slide_id, label))
                continue
            if slide_id not in feature_ids:
                missing_features.append(slide_id)
                if not include_missing_features:
                    continue
            case_value = row.get(case_col, slide_id) if case_col else slide_id
            case_id = strip_slide_suffix(case_value) or slide_id
            rows.append({
                "dir": str(feature_dir),
                "case_id": case_id,
                "slide_id": slide_id,
                "label": label,
            })

    rows.sort(key=lambda item: item["slide_id"])
    return rows, missing_features, bad_labels, feature_ids


def write_csv(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("dir", "case_id", "slide_id", "label"))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset_csv/UBC-OCEAN.csv")
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="UBC-OCEAN metadata CSV, usually Kaggle train.csv with image_id,label columns.",
    )
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-missing-features",
        action="store_true",
        help="Keep metadata rows even if the corresponding .pt feature is missing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.metadata.is_file():
        raise FileNotFoundError("Metadata CSV not found: {}".format(args.metadata))

    rows, missing_features, bad_labels, feature_ids = build_rows(
        args.metadata,
        args.feature_dir,
        include_missing_features=args.include_missing_features,
    )
    if not rows:
        raise ValueError(
            "No usable UBC rows were produced. Check metadata labels and feature ids."
        )

    write_csv(rows, args.output)
    labels = {}
    for row in rows:
        labels[row["label"]] = labels.get(row["label"], 0) + 1

    print("Wrote {}".format(args.output))
    print("Rows: {} | Feature files: {}".format(len(rows), len(feature_ids)))
    print("Label counts: {}".format(", ".join("{}={}".format(k, labels[k]) for k in sorted(labels))))
    if missing_features:
        print("Skipped {} rows without .pt features".format(len(missing_features)))
    if bad_labels:
        print("Skipped {} rows with labels outside {}".format(len(bad_labels), sorted(VALID_LABELS)))


if __name__ == "__main__":
    main()
