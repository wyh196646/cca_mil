#!/usr/bin/env python3
"""Prepare local CAMELYON metadata for FOCUS-style splits.

The original FOCUS splits use CAMELYON16 ids like ``slide_247``. The local
preprocessing writes features with the real slide stems, such as
``normal_087`` or ``test_114``. This script maps the split CSVs back to those
local ids and writes a dataset CSV matching the local ``pt_files`` directory.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_DIR = Path("/data/yuhaowang/WSIFew/processd_wsi/CAMELYON/feature/pt_files")
DEFAULT_MAPPING_CSV = REPO_ROOT / "splits" / "camelyon16_focus_slide_mapping.csv"
DEFAULT_CAMELYON17_CSV = Path("/data/yuhaowang/WSIFew/CAMELYON17/camelyon17_stage_4subtyping.csv")
DEFAULT_OUT_CSV = REPO_ROOT / "dataset_csv" / "camelyon.csv"
DEFAULT_SPLITS_ROOT = REPO_ROOT / "splits"
DEFAULT_SPLIT_DIRS = (
    "camelyon_4shots_10folds",
    "camelyon_8shots_10folds",
    "camelyon_16shots_10folds",
)


def _natural_key(value: str) -> tuple:
    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _load_camelyon16_mapping(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    df = pd.read_csv(path)
    required = {"focus_slide_id", "source_slide_id", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")

    focus_to_source = {
        str(row.focus_slide_id): str(row.source_slide_id)
        for row in df.itertuples(index=False)
    }
    labels = {
        str(row.source_slide_id): str(row.label).strip().lower()
        for row in df.itertuples(index=False)
    }
    return focus_to_source, labels


def _load_camelyon17_labels(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"slide_id", "stage"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")

    labels = {}
    for row in df.itertuples(index=False):
        slide_id = str(row.slide_id)
        stage = str(row.stage).strip().lower()
        labels[slide_id] = "normal" if stage == "negative" else "tumor"
    return labels


def _feature_ids(feature_dir: Path) -> list[str]:
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
    ids = sorted((path.stem for path in feature_dir.glob("*.pt")), key=_natural_key)
    if not ids:
        raise FileNotFoundError(f"No .pt feature files found in {feature_dir}")
    return ids


def _build_dataset_csv(feature_ids: list[str], labels: dict[str, str], feature_dir: Path) -> pd.DataFrame:
    missing = sorted([slide_id for slide_id in feature_ids if slide_id not in labels], key=_natural_key)
    if missing:
        raise KeyError("Missing labels for feature ids: {}".format(", ".join(missing[:20])))

    rows = []
    for slide_id in feature_ids:
        rows.append(
            {
                "dir": str(feature_dir),
                "case_id": slide_id,
                "slide_id": slide_id,
                "label": labels[slide_id],
            }
        )
    return pd.DataFrame(rows)


def _backup_split_dir(split_dir: Path) -> Path:
    backup_dir = split_dir.with_name(split_dir.name + "_focus_ids")
    if not backup_dir.exists():
        shutil.copytree(split_dir, backup_dir)
    return backup_dir


def _map_split_csv(path: Path, focus_to_source: dict[str, str], valid_ids: set[str]) -> tuple[int, list[str]]:
    df = pd.read_csv(path, index_col=0)
    mapped_count = 0
    missing = []

    for column in ("train", "val", "test"):
        if column not in df.columns:
            continue
        values = []
        for value in df[column].tolist():
            if pd.isna(value):
                values.append(value)
                continue
            slide_id = str(value)
            mapped = focus_to_source.get(slide_id, slide_id)
            if mapped != slide_id:
                mapped_count += 1
            if mapped not in valid_ids:
                missing.append(mapped)
            values.append(mapped)
        df[column] = values

    df.to_csv(path)
    return mapped_count, sorted(set(missing), key=_natural_key)


def prepare(args: argparse.Namespace) -> None:
    focus_to_source, c16_labels = _load_camelyon16_mapping(args.camelyon16_mapping)
    c17_labels = _load_camelyon17_labels(args.camelyon17_labels)
    labels = {}
    labels.update(c16_labels)
    labels.update(c17_labels)

    ids = _feature_ids(args.feature_dir)
    dataset = _build_dataset_csv(ids, labels, args.feature_dir)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.out_csv, index=False)

    valid_ids = set(dataset["slide_id"].astype(str))
    print(f"Wrote dataset CSV: {args.out_csv} ({len(dataset)} rows)")
    print("Dataset labels:", dataset["label"].value_counts().to_dict())

    for split_name in args.split_dirs:
        split_dir = args.splits_root / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        backup_dir = _backup_split_dir(split_dir)
        total_mapped = 0
        all_missing = []
        for csv_path in sorted(split_dir.glob("splits_*.csv"), key=lambda p: _natural_key(p.stem)):
            mapped_count, missing = _map_split_csv(csv_path, focus_to_source, valid_ids)
            total_mapped += mapped_count
            all_missing.extend(missing)
        if all_missing:
            raise FileNotFoundError(
                f"{split_dir} contains ids without .pt features after mapping: "
                + ", ".join(sorted(set(all_missing), key=_natural_key)[:20])
            )
        print(f"Mapped {split_dir} ({total_mapped} cells); backup: {backup_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--camelyon16_mapping", type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--camelyon17_labels", type=Path, default=DEFAULT_CAMELYON17_CSV)
    parser.add_argument("--out_csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--splits_root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--split_dirs", nargs="+", default=list(DEFAULT_SPLIT_DIRS))
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
