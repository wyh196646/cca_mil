#!/usr/bin/env python3
"""Build the CAMELYON16 slide_N mapping used by FOCUS-style splits.

FOCUS stores the CAMELYON16 part as slide_1 ... slide_399. For the local
CAMELYON16 release used here, those ids correspond to:

1. normal_*.tif in numeric order
2. tumor_*.tif in numeric order
3. test_*.tif in numeric order

The script writes a CSV mapping these FOCUS ids back to the original files and
optionally verifies the labels against the FOCUS/PathARK camelyon.csv table.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMELYON16_DIR = Path("/data2/yuhaowang/WSIFew/CAMELYON16")
DEFAULT_FOCUS_CSV = Path("/home/yuhaowang/project/PathARK/data/dataset_csv/camelyon.csv")
DEFAULT_OUT_CSV = REPO_ROOT / "splits" / "camelyon16_focus_slide_mapping.csv"


def _numeric_key(path: Path) -> int:
    match = re.search(r"_(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse numeric id from {path.name}")
    return int(match.group(1))


def _collect_ordered_slides(camelyon16_dir: Path) -> list[Path]:
    ordered: list[Path] = []
    for prefix in ("normal", "tumor", "test"):
        slides = sorted(camelyon16_dir.rglob(f"{prefix}_*.tif"), key=_numeric_key)
        if not slides:
            raise FileNotFoundError(f"No {prefix}_*.tif files found under {camelyon16_dir}")
        ordered.extend(slides)
    return ordered


def _load_test_labels(camelyon16_dir: Path) -> dict[str, str]:
    label_csv = camelyon16_dir / "camelyon16_test_cancer.csv"
    if not label_csv.is_file():
        raise FileNotFoundError(f"Missing CAMELYON16 test label CSV: {label_csv}")

    df = pd.read_csv(label_csv, encoding="utf-8-sig")
    expected = {"slide_id", "cancer"}
    if not expected.issubset(df.columns):
        raise ValueError(f"{label_csv} must contain columns {sorted(expected)}")

    return {
        str(row.slide_id): str(row.cancer).strip().lower()
        for row in df.itertuples(index=False)
    }


def _label_for_slide(path: Path, test_labels: dict[str, str]) -> str:
    slide_id = path.stem
    if slide_id.startswith("normal_"):
        return "normal"
    if slide_id.startswith("tumor_"):
        return "tumor"
    if slide_id.startswith("test_"):
        if slide_id not in test_labels:
            raise KeyError(f"No test label found for {slide_id}")
        label = test_labels[slide_id]
        if label not in {"normal", "tumor"}:
            raise ValueError(f"Unexpected label for {slide_id}: {label}")
        return label
    raise ValueError(f"Unknown CAMELYON16 slide prefix: {slide_id}")


def _focus_camelyon16_labels(focus_csv: Path) -> dict[str, str]:
    if not focus_csv.is_file():
        return {}

    df = pd.read_csv(focus_csv)
    required = {"dir", "slide_id", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"{focus_csv} must contain columns {sorted(required)}")

    c16 = df[df["dir"].astype(str).str.contains("CAMELYON16", case=False, na=False)]
    return {
        str(row.slide_id): str(row.label).strip().lower()
        for row in c16.itertuples(index=False)
    }


def build_mapping(camelyon16_dir: Path, focus_csv: Path | None) -> pd.DataFrame:
    camelyon16_dir = camelyon16_dir.resolve()
    ordered_slides = _collect_ordered_slides(camelyon16_dir)
    test_labels = _load_test_labels(camelyon16_dir)
    focus_labels = _focus_camelyon16_labels(focus_csv) if focus_csv is not None else {}

    rows = []
    for index, slide_path in enumerate(ordered_slides, start=1):
        focus_slide_id = f"slide_{index}"
        label = _label_for_slide(slide_path, test_labels)
        focus_label = focus_labels.get(focus_slide_id)
        rows.append(
            {
                "slide_id": focus_slide_id,
                "focus_slide_id": focus_slide_id,
                "source_dataset": "CAMELYON16",
                "source_slide_id": slide_path.stem,
                "source_filename": slide_path.name,
                "source_path": str(slide_path),
                "label": label,
                "focus_label": focus_label,
                "label_match": "" if focus_label is None else label == focus_label,
            }
        )

    return pd.DataFrame(rows)


def _print_queries(mapping: pd.DataFrame, queries: list[str]) -> None:
    if not queries:
        return

    by_id = mapping.set_index("focus_slide_id", drop=False)
    for query in queries:
        if query not in by_id.index:
            print(f"{query}: not found")
            continue
        row = by_id.loc[query]
        print(
            f"{query} -> {row.source_filename} "
            f"({row.label}), path={row.source_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camelyon16_dir", type=Path, default=DEFAULT_CAMELYON16_DIR)
    parser.add_argument("--focus_csv", type=Path, default=DEFAULT_FOCUS_CSV)
    parser.add_argument("--out_csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--query", nargs="*", default=[], help="FOCUS ids to print, e.g. slide_247")
    parser.add_argument("--no_strict", action="store_true", help="Do not fail on FOCUS label mismatches")
    args = parser.parse_args()

    mapping = build_mapping(args.camelyon16_dir, args.focus_csv)
    if "label_match" in mapping.columns and mapping["label_match"].ne("").any():
        mismatches = mapping[mapping["label_match"] == False]  # noqa: E712
        if not mismatches.empty:
            print(mismatches[["focus_slide_id", "source_slide_id", "label", "focus_label"]])
            if not args.no_strict:
                raise SystemExit(f"Found {len(mismatches)} label mismatches against {args.focus_csv}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.out_csv, index=False)

    counts = mapping["label"].value_counts().to_dict()
    print(f"Wrote {len(mapping)} rows to {args.out_csv}")
    print(f"Label counts: {counts}")
    _print_queries(mapping, args.query)


if __name__ == "__main__":
    main()
