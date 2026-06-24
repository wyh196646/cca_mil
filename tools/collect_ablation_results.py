#!/usr/bin/env python3
# coding=utf-8
"""Collect ablation result.csv files into one analysis-ready CSV."""

import argparse
import ast
import csv
import re
from pathlib import Path


DEFAULT_ROOT = Path("/data2/yuhaowang/cca-mil-result/results")
DEFAULT_OUTPUT = DEFAULT_ROOT / "ablation_results_all.csv"
DEFAULT_LIBRA_SUMMARY = DEFAULT_ROOT / "Libra-MIL" / "summary_results.csv"
DEFAULT_BEST_OUTPUT = DEFAULT_ROOT / "CCA_MIL_best_vs_Libra.csv"

PARAM_COLUMNS = [
    "lr",
    "num_visual_prototypes",
    "proto_tau",
    "ot_epsilon",
    "sinkhorn_iter",
    "uot_rho_a",
    "uot_rho_b",
    "concept_pooling",
    "lambda_contrast",
    "lambda_div",
    "contrast_tau",
    "common_concept_weight",
    "train_concept_prompt",
    "concept_prompt_n_ctx",
    "concept_prompt_template_count",
    "max_train_patches",
    "max_eval_patches",
    "concept_logit_weight",
    "concept_logit_tau",
    "model_type",
    "task",
    "seed",
    "num_splits",
    "k_start",
    "k_end",
    "split_dir",
    "csv_path",
    "concept_bank_path",
]


def parse_result_csv(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    metrics = {}
    for row in rows:
        metric_name = str(row.get("metric", "")).strip().lower()
        if not metric_name:
            continue
        suffix = "mean" if metric_name == "mean" else "std" if metric_name in {"std", "var"} else metric_name
        for key, value in row.items():
            if key == "metric" or value in (None, ""):
                continue
            try:
                metrics["{}_{}".format(key, suffix)] = float(value)
            except ValueError:
                metrics["{}_{}".format(key, suffix)] = value
    return metrics


def result_csv_is_complete(path):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        metrics = parse_result_csv(path)
    except (OSError, csv.Error, UnicodeDecodeError):
        return False
    required_any = (
        "val_auc_mean",
        "val_f1_mean",
        "val_acc_mean",
        "test_auc_mean",
        "test_f1_mean",
        "test_acc_mean",
    )
    return any(key in metrics for key in required_any)


def find_experiment_file(result_path):
    exp_files = sorted(result_path.parent.glob("experiment_*.txt"))
    return exp_files[0] if exp_files else None


def parse_experiment_file(path):
    if path is None or not path.is_file():
        return {}
    text = path.read_text(errors="replace").strip()
    if not text:
        return {}
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    return value if isinstance(value, dict) else {}


def stringify(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return repr(value)
    return value


def parse_seed_from_dir(name):
    match = re.search(r"_s(\d+)$", name)
    return int(match.group(1)) if match else ""


def parse_shots(value):
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    match = re.search(r"(\d+)shots", text)
    return int(match.group(1)) if match else ""


def infer_path_metadata(result_path, root):
    rel_dir = result_path.parent.relative_to(root)
    parts = rel_dir.parts
    row = {
        "relative_dir": str(rel_dir),
        "experiment_dir": str(result_path.parent),
        "result_csv": str(result_path),
        "result_file": result_path.name,
        "exp_dir_name": result_path.parent.name,
        "exp_code_from_dir": re.sub(r"_s\d+$", "", result_path.parent.name),
        "seed_from_dir": parse_seed_from_dir(result_path.parent.name),
        "sweep_root": "",
        "sweep_name": "",
        "dataset": "",
        "shots": "",
        "shots_num": "",
        "axis": "",
        "setting": "",
        "param_id": "",
    }

    if len(parts) >= 7 and parts[0] == "AOT_MIL_sweeps":
        row.update({
            "sweep_root": parts[0],
            "sweep_name": parts[1],
            "dataset": parts[2],
            "shots": parts[3],
            "shots_num": parse_shots(parts[3]),
            "axis": parts[4],
            "setting": parts[5],
            "param_id": "{}/{}".format(parts[4], parts[5]),
        })
    elif len(parts) >= 6 and parts[0] == "AOT_MIL_sweeps":
        row.update({
            "sweep_root": parts[0],
            "sweep_name": parts[1],
            "dataset": parts[2],
            "shots": parts[3],
            "shots_num": parse_shots(parts[3]),
            "param_id": parts[4],
        })
    elif len(parts) >= 3:
        row.update({
            "sweep_root": parts[0],
            "dataset": parts[0],
            "param_id": parts[-2] if len(parts) >= 2 else "",
        })
    return row


def collect(root, pattern, only_complete=True):
    result_paths = sorted(path for path in root.rglob(pattern) if path.is_file())
    rows = []
    for result_path in result_paths:
        if result_path.name.startswith("summary"):
            continue
        if only_complete and not result_csv_is_complete(result_path):
            continue

        exp_file = find_experiment_file(result_path)
        exp = parse_experiment_file(exp_file)
        row = infer_path_metadata(result_path, root)
        row["experiment_file"] = str(exp_file) if exp_file else ""

        experiment = exp.get("experiment")
        row["exp_code"] = experiment if experiment else row["exp_code_from_dir"]

        for key in PARAM_COLUMNS:
            row[key] = stringify(exp.get(key, ""))
        if not row["seed"]:
            row["seed"] = row["seed_from_dir"]
        if not row["shots_num"]:
            row["shots_num"] = parse_shots(row.get("split_dir", "")) or parse_shots(row["exp_code"])

        row.update(parse_result_csv(result_path))
        rows.append(row)
    return rows


def filter_rows(rows, datasets, shots, runs):
    dataset_set = {item.strip() for item in datasets.split(",") if item.strip()} if datasets else set()
    shot_set = {int(item.strip()) for item in shots.split(",") if item.strip()} if shots else set()
    run_set = {item.strip() for item in runs.split(",") if item.strip()} if runs else set()
    out = []
    for row in rows:
        if dataset_set and row.get("dataset") not in dataset_set:
            continue
        if shot_set and row.get("shots_num") not in shot_set:
            continue
        if run_set and row.get("sweep_name") not in run_set:
            continue
        out.append(row)
    return out


def sort_rows(rows, metric):
    def key_fn(row):
        dataset = str(row.get("dataset", ""))
        shots = row.get("shots_num")
        try:
            shot_value = int(shots)
        except (TypeError, ValueError):
            shot_value = 10**9
        value = row.get(metric)
        try:
            metric_value = -float(value)
        except (TypeError, ValueError):
            metric_value = float("inf")
        return dataset, shot_value, metric_value

    return sorted(rows, key=key_fn)


def as_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_dataset_name(name):
    mapping = {
        "tcga": "tcga_nsclc",
        "tcga_nsclc": "tcga_nsclc",
        "LUAD_LUSC": "tcga_nsclc",
        "camelyon": "camelyon",
        "ubc": "ubc_ocean",
        "ubc_ocean": "ubc_ocean",
        "UBC-OCEAN": "ubc_ocean",
        "rcc": "tcga_rcc",
        "tcga_rcc": "tcga_rcc",
        "TCGA_RCC": "tcga_rcc",
    }
    return mapping.get(str(name), str(name))


def read_libra_summary(path):
    path = Path(path)
    if not path.is_file():
        return {}
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            dataset = normalize_dataset_name(row.get("dataset", ""))
            shots = parse_shots(row.get("shots", ""))
            if not dataset or shots == "":
                continue
            out[(dataset, int(shots))] = row
    return out


def best_rows_by_group(rows, metric):
    groups = {}
    for row in rows:
        dataset = normalize_dataset_name(row.get("dataset", ""))
        shots = row.get("shots_num")
        try:
            shots = int(shots)
        except (TypeError, ValueError):
            continue
        groups.setdefault((dataset, shots), []).append(row)

    best = []
    for key, group_rows in sorted(groups.items()):
        best_row = max(
            group_rows,
            key=lambda row: (
                as_float(row.get(metric), -1.0),
                as_float(row.get("val_f1_mean"), -1.0),
                -as_float(row.get("val_auc_std"), 1e9),
            ),
        )
        best.append(best_row)
    return best


def write_best_vs_libra(rows, libra_summary, output, metric):
    libra = read_libra_summary(libra_summary)
    best_rows = best_rows_by_group(rows, metric)
    fields = [
        "dataset",
        "shots_num",
        "selection_metric",
        "cca_param_id",
        "cca_axis",
        "cca_setting",
        "cca_exp_code",
        "cca_result_csv",
        "cca_val_auc_mean",
        "cca_val_f1_mean",
        "cca_test_auc_mean",
        "cca_test_auc_std",
        "cca_test_f1_mean",
        "cca_test_f1_std",
        "cca_test_acc_mean",
        "cca_test_acc_std",
        "libra_num_folds",
        "libra_test_auc_mean",
        "libra_test_auc_std",
        "libra_test_f1_mean",
        "libra_test_f1_std",
        "libra_test_acc_mean",
        "libra_test_acc_std",
        "delta_test_auc",
        "delta_test_f1",
        "delta_test_acc",
        "lr",
        "num_visual_prototypes",
        "proto_tau",
        "ot_epsilon",
        "sinkhorn_iter",
        "uot_rho_a",
        "uot_rho_b",
        "concept_pooling",
        "lambda_contrast",
        "lambda_div",
        "contrast_tau",
        "train_concept_prompt",
        "concept_prompt_n_ctx",
        "concept_prompt_template_count",
        "max_train_patches",
        "max_eval_patches",
        "concept_logit_weight",
        "concept_logit_tau",
    ]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for row in best_rows:
        dataset = normalize_dataset_name(row.get("dataset", ""))
        shots = int(row.get("shots_num"))
        libra_row = libra.get((dataset, shots), {})
        cca_auc = as_float(row.get("test_auc_mean"))
        cca_f1 = as_float(row.get("test_f1_mean"))
        cca_acc = as_float(row.get("test_acc_mean"))
        libra_auc = as_float(libra_row.get("test_auc_mean"))
        libra_f1 = as_float(libra_row.get("test_f1_mean"))
        libra_acc = as_float(libra_row.get("test_acc_mean"))
        out = {
            "dataset": dataset,
            "shots_num": shots,
            "selection_metric": metric,
            "cca_param_id": row.get("param_id", ""),
            "cca_axis": row.get("axis", ""),
            "cca_setting": row.get("setting", ""),
            "cca_exp_code": row.get("exp_code", ""),
            "cca_result_csv": row.get("result_csv", ""),
            "cca_val_auc_mean": row.get("val_auc_mean", ""),
            "cca_val_f1_mean": row.get("val_f1_mean", ""),
            "cca_test_auc_mean": row.get("test_auc_mean", ""),
            "cca_test_auc_std": row.get("test_auc_std", ""),
            "cca_test_f1_mean": row.get("test_f1_mean", ""),
            "cca_test_f1_std": row.get("test_f1_std", ""),
            "cca_test_acc_mean": row.get("test_acc_mean", ""),
            "cca_test_acc_std": row.get("test_acc_std", ""),
            "libra_num_folds": libra_row.get("num_folds", ""),
            "libra_test_auc_mean": libra_row.get("test_auc_mean", ""),
            "libra_test_auc_std": libra_row.get("test_auc_std", ""),
            "libra_test_f1_mean": libra_row.get("test_f1_mean", ""),
            "libra_test_f1_std": libra_row.get("test_f1_std", ""),
            "libra_test_acc_mean": libra_row.get("test_acc_mean", ""),
            "libra_test_acc_std": libra_row.get("test_acc_std", ""),
            "delta_test_auc": cca_auc - libra_auc,
            "delta_test_f1": cca_f1 - libra_f1,
            "delta_test_acc": cca_acc - libra_acc,
        }
        for key in fields:
            if key not in out and key in row:
                out[key] = row[key]
        out_rows.append(out)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    return out_rows


def write_csv(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fixed_columns = [
        "sweep_name",
        "dataset",
        "shots",
        "shots_num",
        "axis",
        "setting",
        "param_id",
        "exp_code",
        "seed",
        "val_auc_mean",
        "val_auc_std",
        "val_f1_mean",
        "val_f1_std",
        "val_acc_mean",
        "val_acc_std",
        "test_auc_mean",
        "test_auc_std",
        "test_f1_mean",
        "test_f1_std",
        "test_acc_mean",
        "test_acc_std",
    ]
    columns = []
    for column in fixed_columns + PARAM_COLUMNS:
        if column not in columns:
            columns.append(column)
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser(description="Collect all ablation result.csv files")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pattern", default="result*.csv")
    parser.add_argument("--sort-by", default="test_auc_mean")
    parser.add_argument("--datasets", default="", help="Optional comma list, e.g. tcga,camelyon,ubc")
    parser.add_argument("--shots", default="", help="Optional comma list, e.g. 1,4,16")
    parser.add_argument("--runs", default="", help="Optional comma list of sweep run names")
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--print-top", type=int, default=10)
    parser.add_argument("--libra-summary", type=Path, default=DEFAULT_LIBRA_SUMMARY)
    parser.add_argument("--best-output", type=Path, default=DEFAULT_BEST_OUTPUT)
    parser.add_argument("--no-best-vs-libra", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if not args.root.is_dir():
        write_csv([], args.output)
        print("Result root not found: {}".format(args.root))
        print("Wrote empty summary: {}".format(args.output))
        return

    rows = collect(args.root, args.pattern, only_complete=not args.include_incomplete)
    rows = filter_rows(rows, args.datasets, args.shots, args.runs)
    rows = sort_rows(rows, args.sort_by)
    write_csv(rows, args.output)
    best_rows = []
    if not args.no_best_vs_libra:
        best_rows = write_best_vs_libra(rows, args.libra_summary, args.best_output, args.sort_by)

    print("Collected {} complete result files".format(len(rows)))
    print("Wrote {}".format(args.output))
    if not args.no_best_vs_libra:
        print("Wrote best-vs-Libra table with {} rows: {}".format(len(best_rows), args.best_output))
    if args.print_top > 0 and rows:
        print("Top {} by {}:".format(min(args.print_top, len(rows)), args.sort_by))
        for idx, row in enumerate(sorted(rows, key=lambda r: float(r.get(args.sort_by, -1) or -1), reverse=True)[: args.print_top], start=1):
            print(
                "{:>2}. dataset={} shots={} param={} val_auc={} test_auc={} exp={}".format(
                    idx,
                    row.get("dataset", ""),
                    row.get("shots", ""),
                    row.get("param_id", ""),
                    row.get("val_auc_mean", ""),
                    row.get("test_auc_mean", ""),
                    row.get("exp_code", ""),
                )
            )


if __name__ == "__main__":
    main()
