#!/usr/bin/env python3
"""Audit WSI preprocessing completeness.

The script compares raw WSI files, patch coordinate h5 files, feature pt files,
dataset CSVs, and split CSVs by slide id stem. It writes missing-id CSVs so the
preprocessing queue can be checked before training starts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KNOWN_SUFFIXES = (
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".mrxs",
    ".h5",
    ".pt",
)


def _normalise_exts(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    exts = []
    for ext in value.split(","):
        ext = ext.strip()
        if not ext:
            continue
        exts.append(ext if ext.startswith(".") else "." + ext)
    return tuple(exts)


def _stem(value) -> str:
    name = Path(str(value)).name
    lower = name.lower()
    for suffix in KNOWN_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _scan_files(directory: Path | None, exts: tuple[str, ...]) -> set[str]:
    if directory is None or not directory.is_dir():
        return set()
    if not exts:
        return {_stem(path.name) for path in directory.iterdir() if path.is_file()}
    return {
        _stem(path.name)
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {ext.lower() for ext in exts}
    }


def _load_dataset_ids(csv_path: Path | None) -> set[str]:
    if csv_path is None or not csv_path.is_file():
        return set()
    df = pd.read_csv(csv_path)
    if "slide_id" not in df.columns:
        raise KeyError(f"{csv_path} must contain a slide_id column")
    return {_stem(value) for value in df["slide_id"].dropna().astype(str)}


def _load_split_ids(split_dirs: list[Path]) -> set[str]:
    ids: set[str] = set()
    for split_dir in split_dirs:
        if not split_dir.is_dir():
            continue
        for csv_path in sorted(split_dir.glob("splits_*.csv")):
            df = pd.read_csv(csv_path)
            for key in ["train", "val", "test"]:
                if key in df.columns:
                    ids.update(_stem(value) for value in df[key].dropna().astype(str))
    return ids


def _write_missing(out_dir: Path, name: str, ids: set[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    pd.DataFrame({"slide_id": sorted(ids)}).to_csv(path, index=False)
    print(f"{name}: {len(ids)} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--processed_dir", type=Path, required=True)
    parser.add_argument("--feat_dir", type=Path, default=None)
    parser.add_argument("--dataset_csv", type=Path, default=None)
    parser.add_argument("--split_dir", type=Path, nargs="*", default=[])
    parser.add_argument("--slide_exts", type=str, default=".svs,.tif,.ndpi,.mrxs")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if split ids miss features")
    args = parser.parse_args()

    raw_ids = _scan_files(args.raw_dir, _normalise_exts(args.slide_exts))
    patch_ids = _scan_files(args.processed_dir / "patches", (".h5",))
    feat_dir = args.feat_dir if args.feat_dir is not None else args.processed_dir / "feature"
    feature_ids = _scan_files(feat_dir / "pt_files", (".pt",))
    dataset_ids = _load_dataset_ids(args.dataset_csv)
    split_ids = _load_split_ids(args.split_dir)

    out_dir = args.out_dir if args.out_dir is not None else args.processed_dir / "audit"

    print("Counts")
    print(f"  raw:      {len(raw_ids)}")
    print(f"  patch:    {len(patch_ids)}")
    print(f"  feature:  {len(feature_ids)}")
    if dataset_ids:
        print(f"  dataset:  {len(dataset_ids)}")
    if split_ids:
        print(f"  splits:   {len(split_ids)}")

    checks = {
        "raw_missing_patch": raw_ids - patch_ids,
        "patch_missing_feature": patch_ids - feature_ids,
        "feature_without_patch": feature_ids - patch_ids,
    }
    if dataset_ids:
        checks.update({
            "dataset_missing_raw": dataset_ids - raw_ids,
            "dataset_missing_patch": dataset_ids - patch_ids,
            "dataset_missing_feature": dataset_ids - feature_ids,
        })
    if split_ids:
        checks.update({
            "split_missing_raw": split_ids - raw_ids,
            "split_missing_patch": split_ids - patch_ids,
            "split_missing_feature": split_ids - feature_ids,
        })

    for name, ids in checks.items():
        _write_missing(out_dir, name, ids)

    if args.strict and split_ids and (split_ids - feature_ids):
        raise SystemExit("Split ids are missing feature .pt files")


if __name__ == "__main__":
    main()
