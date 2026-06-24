#!/usr/bin/env python3
"""Prepare TCGA-RCC metadata from the wide result.csv manifest.

The Libra-MIL protocol keeps primary tumor diagnostic WSIs. By default this
script keeps sample type 01 and requires a matching .pt feature, which is the
format consumed by cca_mil.
"""

import argparse
import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_CSV = ROOT / "result.csv"
DEFAULT_OUTPUT_CSV = ROOT / "dataset_csv" / "RCC.csv"
DEFAULT_DROPPED_CSV = ROOT / "dataset_csv" / "RCC_dropped.csv"
DEFAULT_FEATURE_DIR = Path("/data2/yuhaowang/WSIFew/processd_wsi/TCGA-RCC/feature/pt_files")

LABELS = {
    "TCGA-KICH": "KICH",
    "TCGA-KIRC": "KIRC",
    "TCGA-KIRP": "KIRP",
}


def parse_csv_list(value):
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def slide_stem(filename):
    return Path(str(filename).strip()).stem


def sample_type(slide_id):
    parts = str(slide_id).split("-")
    if len(parts) < 4:
        return ""
    return parts[3][:2]


def case_id(slide_id):
    return "-".join(str(slide_id).split("-")[:3])


def feature_exists(slide_id, feature_dir, feature_ext):
    if not feature_dir:
        return True
    return (feature_dir / "{}.{}".format(slide_id, feature_ext.lstrip("."))).is_file()


def iter_rows(result_csv):
    df = pd.read_csv(result_csv, dtype=str, encoding="utf-8-sig")
    missing_columns = sorted(set(LABELS) - set(df.columns))
    if missing_columns:
        raise ValueError("{} is missing columns: {}".format(result_csv, ", ".join(missing_columns)))

    for source_column, label in LABELS.items():
        values = df[source_column].dropna().astype(str).str.strip()
        values = values[values != ""]
        for source_file in values:
            slide_id = slide_stem(source_file)
            yield {
                "case_id": case_id(slide_id),
                "slide_id": slide_id,
                "label": label,
                "source_file": source_file,
                "source_column": source_column,
                "sample_type": sample_type(slide_id),
            }


def prepare_metadata(args):
    result_csv = Path(args.result_csv)
    output_csv = Path(args.output_csv)
    dropped_csv = Path(args.dropped_csv)
    feature_dir = Path(args.feature_dir) if args.feature_dir else None
    keep_sample_types = set(parse_csv_list(args.sample_types))

    kept = []
    dropped = []
    seen = set()
    for row in iter_rows(result_csv):
        reasons = []
        if row["slide_id"] in seen:
            reasons.append("duplicate_slide_id")
        if keep_sample_types and row["sample_type"] not in keep_sample_types:
            reasons.append("sample_type_{}".format(row["sample_type"] or "unknown"))
        if not feature_exists(row["slide_id"], feature_dir, args.feature_ext):
            reasons.append("missing_{}_feature".format(args.feature_ext.lstrip(".")))

        if reasons:
            dropped.append(dict(row, drop_reason=";".join(reasons)))
            continue

        seen.add(row["slide_id"])
        kept.append(row)

    kept.sort(key=lambda item: (item["label"], item["slide_id"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(kept, columns=["case_id", "slide_id", "label", "source_file"]).to_csv(output_csv, index=False)

    if dropped:
        pd.DataFrame(dropped).to_csv(dropped_csv, index=False)
    elif dropped_csv.exists():
        dropped_csv.unlink()

    counts = pd.Series([row["label"] for row in kept], dtype=str).value_counts().sort_index().to_dict()
    summary_path = output_csv.with_name(output_csv.stem + "_summary.csv")
    with summary_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["label", "count"])
        for label in ("KICH", "KIRC", "KIRP"):
            writer.writerow([label, counts.get(label, 0)])
        writer.writerow(["total", len(kept)])
        writer.writerow(["dropped", len(dropped)])

    print("Wrote {}".format(output_csv))
    print("Wrote {}".format(summary_path))
    if dropped:
        print("Wrote {}".format(dropped_csv))
    print("Kept counts: {}".format(counts))
    print("Dropped: {}".format(len(dropped)))


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare local TCGA-RCC metadata CSV")
    parser.add_argument("--result-csv", default=str(DEFAULT_RESULT_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--dropped-csv", default=str(DEFAULT_DROPPED_CSV))
    parser.add_argument("--sample-types", default="01", help="Comma list of TCGA sample type codes to keep")
    parser.add_argument("--feature-dir", default=str(DEFAULT_FEATURE_DIR))
    parser.add_argument("--feature-ext", default="pt")
    return parser.parse_args()


def main():
    prepare_metadata(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
