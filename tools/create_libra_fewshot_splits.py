#!/usr/bin/env python3
# coding=utf-8
"""Create Libra-MIL style stratified 5-fold few-shot splits.

Protocol:
  - stratified 5-fold cross validation;
  - for fold i, test uses fold i and validation uses fold (i + 1) % k;
  - training samples are drawn as M shots per class from the remaining folds.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHOTS = "1,4,16"
DEFAULT_FOLDS = 5


@dataclass(frozen=True)
class SplitDataset:
    key: str
    prefix: str
    csv_path: Path
    label_order: tuple


DATASETS = {
    "tcga": SplitDataset(
        key="tcga",
        prefix="LUAD_LUSC",
        csv_path=ROOT / "dataset_csv" / "LUAD_LUSC.csv",
        label_order=("LUAD", "LUSC"),
    ),
    "camelyon": SplitDataset(
        key="camelyon",
        prefix="camelyon",
        csv_path=ROOT / "dataset_csv" / "camelyon.csv",
        label_order=("normal", "tumor"),
    ),
    "ubc": SplitDataset(
        key="ubc",
        prefix="UBC-OCEAN",
        csv_path=ROOT / "dataset_csv" / "UBC-OCEAN.csv",
        label_order=("CC", "HGSC", "LGSC", "EC", "MC"),
    ),
    "rcc": SplitDataset(
        key="rcc",
        prefix="TCGA_RCC",
        csv_path=ROOT / "dataset_csv" / "RCC.csv",
        label_order=("KICH", "KIRC", "KIRP"),
    ),
}


def parse_csv_list(value, cast=str):
    if value is None or value == "":
        return []
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def split_series(values):
    max_len = max((len(v) for v in values.values()), default=0)
    rows = {}
    for key, value in values.items():
        padded = list(value) + [""] * (max_len - len(value))
        rows[key] = padded
    return pd.DataFrame(rows, columns=["train", "val", "test"])


def make_bool_df(all_ids, split_values):
    rows = []
    for slide_id in all_ids:
        rows.append({
            "slide_id": slide_id,
            "train": slide_id in split_values["train"],
            "val": slide_id in split_values["val"],
            "test": slide_id in split_values["test"],
        })
    return pd.DataFrame(rows).set_index("slide_id")


def make_descriptor(df, split_values, label_order):
    rows = []
    label_by_slide = dict(zip(df["slide_id"].astype(str), df["label"].astype(str)))
    for label in label_order:
        row = {"label": label}
        for split_name, slide_ids in split_values.items():
            row[split_name] = sum(label_by_slide[str(slide_id)] == label for slide_id in slide_ids)
        rows.append(row)
    return pd.DataFrame(rows).set_index("label")


def check_split(split_values):
    names = ("train", "val", "test")
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = set(split_values[left]).intersection(split_values[right])
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                raise ValueError("{} and {} overlap: {}".format(left, right, examples))


def build_class_chunks(df, label_order, folds, seed):
    rng = np.random.default_rng(seed)
    chunks = {}
    for label in label_order:
        slide_ids = df.loc[df["label"].astype(str) == label, "slide_id"].astype(str).to_numpy()
        if len(slide_ids) < folds:
            raise ValueError("Class {} has {} samples, fewer than {} folds".format(label, len(slide_ids), folds))
        shuffled = slide_ids.copy()
        rng.shuffle(shuffled)
        chunks[label] = [chunk.tolist() for chunk in np.array_split(shuffled, folds)]
    return chunks


def build_split(df, chunks, label_order, shot, fold_idx, folds, seed):
    rng = np.random.default_rng(seed + fold_idx * 1009 + shot * 9173)
    test_fold = fold_idx
    val_fold = (fold_idx + 1) % folds
    split_values = {"train": [], "val": [], "test": []}

    for label in label_order:
        label_chunks = chunks[label]
        split_values["val"].extend(label_chunks[val_fold])
        split_values["test"].extend(label_chunks[test_fold])
        train_pool = []
        for idx, chunk in enumerate(label_chunks):
            if idx not in (val_fold, test_fold):
                train_pool.extend(chunk)
        if len(train_pool) < shot:
            raise ValueError(
                "Class {} has only {} training-pool samples in fold {}, cannot draw {} shots".format(
                    label, len(train_pool), fold_idx, shot
                )
            )
        selected = rng.choice(np.array(train_pool, dtype=object), size=shot, replace=False).tolist()
        split_values["train"].extend(selected)

    for key in split_values:
        split_values[key] = sorted(split_values[key])
    check_split(split_values)
    return split_values


def write_one_dataset(cfg, shots, folds, seed, output_root, overwrite):
    csv_path = Path(cfg.csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError("missing dataset CSV: {}".format(csv_path))

    df = pd.read_csv(csv_path)
    required = {"slide_id", "label"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("{} is missing columns: {}".format(csv_path, ", ".join(missing)))
    df = df.copy()
    df["slide_id"] = df["slide_id"].astype(str)
    df["label"] = df["label"].astype(str)

    unknown = sorted(set(df["label"]) - set(cfg.label_order))
    if unknown:
        raise ValueError("{} has labels not in label_order: {}".format(csv_path, ", ".join(unknown)))

    chunks = build_class_chunks(df, cfg.label_order, folds, seed)
    created = []
    for shot in shots:
        out_dir = output_root / "{}_{}shots_{}folds".format(cfg.prefix, shot, folds)
        if out_dir.exists() and not overwrite:
            raise FileExistsError("{} exists; pass --overwrite to replace files".format(out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_rows = []
        for fold_idx in range(folds):
            split_values = build_split(df, chunks, cfg.label_order, shot, fold_idx, folds, seed)
            split_df = split_series(split_values)
            split_df.to_csv(out_dir / "splits_{}.csv".format(fold_idx))
            make_bool_df(df["slide_id"].tolist(), split_values).to_csv(out_dir / "splits_{}_bool.csv".format(fold_idx))
            descriptor = make_descriptor(df, split_values, cfg.label_order)
            descriptor.to_csv(out_dir / "splits_{}_descriptor.csv".format(fold_idx))

            for label in cfg.label_order:
                row = {"fold": fold_idx, "shot": shot, "label": label}
                row.update(descriptor.loc[label].to_dict())
                summary_rows.append(row)

        with (out_dir / "split_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["fold", "shot", "label", "train", "val", "test"])
            writer.writeheader()
            writer.writerows(summary_rows)

        protocol = {
            "source_csv": str(csv_path),
            "dataset": cfg.key,
            "prefix": cfg.prefix,
            "shots": shot,
            "folds": folds,
            "seed": seed,
            "val_fold": "(fold + 1) % folds",
            "train_sampling": "shot samples per class from folds not used for val/test",
            "label_order": list(cfg.label_order),
        }
        (out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        created.append(out_dir)
    return created


def parse_args():
    parser = argparse.ArgumentParser(description="Create Libra-MIL style few-shot 5-fold splits")
    parser.add_argument("--datasets", default="all", help="Comma list: all,tcga,camelyon,ubc,rcc")
    parser.add_argument("--shots", default=DEFAULT_SHOTS, help="Comma list, default: {}".format(DEFAULT_SHOTS))
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-root", default=str(ROOT / "splits"))
    parser.add_argument("--tcga-csv-path", default=None)
    parser.add_argument("--camelyon-csv-path", default=None)
    parser.add_argument("--ubc-csv-path", default=None)
    parser.add_argument("--rcc-csv-path", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = parse_csv_list(args.datasets)
    if not datasets or datasets == ["all"]:
        datasets = ["tcga", "camelyon", "ubc", "rcc"]

    configs = dict(DATASETS)
    if args.tcga_csv_path:
        configs["tcga"] = SplitDataset("tcga", "LUAD_LUSC", Path(args.tcga_csv_path), DATASETS["tcga"].label_order)
    if args.camelyon_csv_path:
        configs["camelyon"] = SplitDataset("camelyon", "camelyon", Path(args.camelyon_csv_path), DATASETS["camelyon"].label_order)
    if args.ubc_csv_path:
        configs["ubc"] = SplitDataset("ubc", "UBC-OCEAN", Path(args.ubc_csv_path), DATASETS["ubc"].label_order)
    if args.rcc_csv_path:
        configs["rcc"] = SplitDataset("rcc", "TCGA_RCC", Path(args.rcc_csv_path), DATASETS["rcc"].label_order)

    shots = parse_csv_list(args.shots, int)
    output_root = Path(args.output_root)
    all_created = []
    for key in datasets:
        if key not in configs:
            raise ValueError("Unknown dataset '{}'. Choose from {}".format(key, sorted(configs)))
        created = write_one_dataset(configs[key], shots, args.folds, args.seed, output_root, args.overwrite)
        all_created.extend(created)
        for path in created:
            print("[created] {}".format(path))

    print("Created {} split directories.".format(len(all_created)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
