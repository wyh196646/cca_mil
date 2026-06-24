#!/usr/bin/env python3
# coding=utf-8
"""Run resumable AOT-MIL hyperparameter sweeps across datasets and GPUs."""

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("/data2/yuhaowang/cca-mil-result")
DEFAULT_PYTHON = os.environ.get("CCA_MIL_PYTHON", sys.executable)
PARAMETER_KEYS = [
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
]
DEFAULT_COMBO = {
    "lr": 1e-4,
    "num_visual_prototypes": 6,
    "proto_tau": 0.1,
    "ot_epsilon": 0.05,
    "sinkhorn_iter": 20,
    "uot_rho_a": 0.5,
    "uot_rho_b": 0.5,
    "concept_pooling": "attention",
    "lambda_contrast": 0.1,
    "lambda_div": 0.01,
    "contrast_tau": 0.07,
    "common_concept_weight": 0.3,
    "train_concept_prompt": True,
    "concept_prompt_n_ctx": 4,
    "concept_prompt_template_count": 4,
    "max_train_patches": 4096,
    "max_eval_patches": 0,
    "concept_logit_weight": 0.5,
    "concept_logit_tau": 1.0,
}
LEGACY_MISSING_DEFAULTS = {
    "train_concept_prompt": False,
    "concept_prompt_n_ctx": 0,
    "concept_prompt_template_count": 0,
    "max_train_patches": 0,
    "max_eval_patches": 0,
    "concept_logit_weight": 0.0,
    "concept_logit_tau": 1.0,
}


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    name: str
    task: str
    csv_path: Path
    feature_dir: Path
    concept_bank: Path
    split_template: str
    default_visual_prototypes: int


def parse_csv_list(value, cast=str):
    if value is None or value == "":
        return []
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def fmt_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-3:
            text = "{:.0e}".format(value)
            return text.replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")
        return "{:.6g}".format(value)
    return str(value)


def safe_token(value):
    text = fmt_value(value).strip()
    text = re.sub(r"[^A-Za-z0-9._+-]+", "_", text)
    return text.strip("_") or "default"


def param_summary(combo):
    return ",".join("{}={}".format(key, fmt_value(combo[key])) for key in PARAMETER_KEYS)


def canonical_combo(combo):
    """Key by parameters that change the current CCA_MIL computation graph."""
    normalized = {}
    for key in PARAMETER_KEYS:
        if key == "common_concept_weight":
            # The current model stores concept weights but does not use them in
            # forward/loss, so sweeping this creates a no-op duplicate.
            continue
        if key == "contrast_tau" and float(combo.get("lambda_contrast", 0.0) or 0.0) <= 0:
            continue
        if key == "concept_prompt_n_ctx" and not bool(combo.get("train_concept_prompt", False)):
            continue
        if key == "concept_prompt_template_count" and int(combo.get("concept_prompt_template_count", 0) or 0) <= 0:
            continue
        if key == "concept_logit_tau" and float(combo.get("concept_logit_weight", 0.0) or 0.0) <= 0:
            continue
        normalized[key] = combo[key]
    return json.dumps(normalized, sort_keys=True)


def coerce_combo_value(key, value):
    if key == "concept_pooling":
        return str(value)
    if key == "train_concept_prompt":
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if key in {"num_visual_prototypes", "sinkhorn_iter", "concept_prompt_n_ctx", "concept_prompt_template_count", "max_train_patches", "max_eval_patches"}:
        return int(value)
    return float(value)


def make_entry(axis, setting, **updates):
    combo = dict(DEFAULT_COMBO)
    combo.update(updates)
    axis = safe_token(axis)
    setting = safe_token(setting)
    return {
        "axis": axis,
        "setting": setting,
        "tag": "{}_{}".format(axis, setting),
        "param_id": "{}/{}".format(axis, setting),
        "param_summary": param_summary(combo),
        "canonical_key": canonical_combo(combo),
        "combo": combo,
    }


def dedupe_entries(entries):
    deduped = []
    seen = {}
    for entry in entries:
        key = entry["canonical_key"]
        if key in seen:
            print(
                "[dedupe-grid] drop {} because it is equivalent to {}".format(
                    entry["param_id"], seen[key]
                ),
                flush=True,
            )
            continue
        seen[key] = entry["param_id"]
        deduped.append(entry)
    return deduped


def apply_dataset_defaults(cfg, axis, combo):
    combo = dict(combo)
    if (
        axis not in {"visual_prototypes", "num_visual_prototypes"}
        and combo.get("num_visual_prototypes") == DEFAULT_COMBO["num_visual_prototypes"]
    ):
        combo["num_visual_prototypes"] = cfg.default_visual_prototypes
    return combo


def dataset_configs(args):
    tcga_feature = Path(args.tcga_feature_dir or os.environ.get(
        "CCA_MIL_TCGA_FEATURE_DIR",
        "/data2/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature/pt_files",
    ))
    camelyon_feature = Path(args.camelyon_feature_dir or os.environ.get(
        "CCA_MIL_CAMELYON_FEATURE_DIR",
        "/data2/yuhaowang/WSIFew/processd_wsi/CAMELYON/feature/pt_files",
    ))
    ubc_feature = Path(args.ubc_feature_dir or os.environ.get(
        "CCA_MIL_UBC_FEATURE_DIR",
        "/data2/yuhaowang/WSIFew/processd_wsi/UBC-OCEAN/feature/pt_files",
    ))
    ubc_csv = Path(args.ubc_csv_path or os.environ.get(
        "CCA_MIL_UBC_CSV",
        str(ROOT / "dataset_csv" / "UBC-OCEAN.csv"),
    ))
    rcc_feature = Path(args.rcc_feature_dir or os.environ.get(
        "CCA_MIL_RCC_FEATURE_DIR",
        "/data2/yuhaowang/WSIFew/processd_wsi/TCGA-RCC/feature/pt_files",
    ))
    rcc_csv = Path(args.rcc_csv_path or os.environ.get(
        "CCA_MIL_RCC_CSV",
        str(ROOT / "dataset_csv" / "RCC.csv"),
    ))

    return {
        "tcga": DatasetConfig(
            key="tcga",
            name="LUAD_LUSC",
            task="task_tcga_lung_subtyping",
            csv_path=Path(args.tcga_csv_path or ROOT / "dataset_csv" / "LUAD_LUSC.csv"),
            feature_dir=tcga_feature,
            concept_bank=ROOT / "text_prompt" / "concept_bank" / "tcga_nsclc.json",
            split_template="LUAD_LUSC_{shots}shots_5folds",
            default_visual_prototypes=6,
        ),
        "camelyon": DatasetConfig(
            key="camelyon",
            name="camelyon",
            task="task_camelyon_subtyping",
            csv_path=Path(args.camelyon_csv_path or ROOT / "dataset_csv" / "camelyon.csv"),
            feature_dir=camelyon_feature,
            concept_bank=ROOT / "text_prompt" / "concept_bank" / "camelyon.json",
            split_template="camelyon_{shots}shots_5folds",
            default_visual_prototypes=10,
        ),
        "ubc": DatasetConfig(
            key="ubc",
            name="UBC-OCEAN",
            task="task_UBC-OCEAN_subtyping",
            csv_path=ubc_csv,
            feature_dir=ubc_feature,
            concept_bank=ROOT / "text_prompt" / "concept_bank" / "ubc_ocean.json",
            split_template="UBC-OCEAN_{shots}shots_5folds",
            default_visual_prototypes=10,
        ),
        "rcc": DatasetConfig(
            key="rcc",
            name="TCGA_RCC",
            task="task_tcga_rcc_subtyping",
            csv_path=rcc_csv,
            feature_dir=rcc_feature,
            concept_bank=ROOT / "text_prompt" / "concept_bank" / "tcga_rcc.json",
            split_template="TCGA_RCC_{shots}shots_5folds",
            default_visual_prototypes=10,
        ),
    }


def balanced_grid():
    entries = [make_entry("base", "default")]

    # Priority 1: optimization. These usually dominate few-shot stability.
    for value in (2e-5, 5e-5, 2e-4, 5e-4):
        entries.append(make_entry("lr", value, lr=value))

    # Priority 2: the core AOT-MIL representation and transport parameters.
    for value in (8, 32):
        entries.append(make_entry("visual_prototypes", value, num_visual_prototypes=value))
    for value in (0.05, 0.2):
        entries.append(make_entry("proto_tau", value, proto_tau=value))
    for value in (0.03, 0.1):
        entries.append(make_entry("ot_epsilon", value, ot_epsilon=value))
    for rho_a, rho_b in ((0.3, 0.3), (1.0, 1.0)):
        entries.append(
            make_entry(
                "uot_rho",
                "{}_{}".format(fmt_value(rho_a), fmt_value(rho_b)),
                uot_rho_a=rho_a,
                uot_rho_b=rho_b,
            )
        )

    # Priority 3: loss ablations. Each setting changes the actual loss terms.
    entries.extend([
        make_entry("loss", "ce_only", lambda_contrast=0.0, lambda_div=0.0),
        make_entry("loss", "contrast_only", lambda_contrast=0.1, lambda_div=0.0),
        make_entry("loss", "weak_contrast", lambda_contrast=0.05, lambda_div=0.01),
        make_entry("loss", "strong_contrast", lambda_contrast=0.2, lambda_div=0.01),
        make_entry("loss", "strong_div", lambda_contrast=0.1, lambda_div=0.02),
    ])

    # Priority 4: new CCA-MIL knobs that directly affect speed/few-shot stability.
    entries.extend([
        make_entry("prompt", "frozen", train_concept_prompt=False, concept_prompt_n_ctx=0),
        make_entry("prompt_ctx", 2, concept_prompt_n_ctx=2),
        make_entry("prompt_ctx", 8, concept_prompt_n_ctx=8),
        make_entry("prompt_templates", 1, concept_prompt_template_count=1),
        make_entry("prompt_templates", 8, concept_prompt_template_count=8),
        make_entry("patch_budget", 2048, max_train_patches=2048),
        make_entry("patch_budget", 8192, max_train_patches=8192),
        make_entry("concept_logits", "off", concept_logit_weight=0.0),
        make_entry("concept_logits", "strong", concept_logit_weight=1.0),
    ])

    return dedupe_entries(entries)


def smoke_grid():
    entries = [
        make_entry("base", "default"),
        make_entry("lr", 5e-5, lr=5e-5),
        make_entry("lr", 2e-4, lr=2e-4),
        make_entry("loss", "ce_only", lambda_contrast=0.0, lambda_div=0.0),
        make_entry("prompt", "frozen", train_concept_prompt=False, concept_prompt_n_ctx=0),
        make_entry("patch_budget", 2048, max_train_patches=2048),
        make_entry("concept_logits", "off", concept_logit_weight=0.0),
    ]
    return dedupe_entries(entries)


def extended_grid():
    entries = list(balanced_grid())
    entries.extend([
        make_entry("sinkhorn_iter", 30, sinkhorn_iter=30),
        make_entry("sinkhorn_iter", 75, sinkhorn_iter=75),
        make_entry("concept_pooling", "mean", concept_pooling="mean"),
        make_entry("concept_pooling", "learnable", concept_pooling="learnable"),
        make_entry("contrast_tau", 0.05, contrast_tau=0.05),
        make_entry("contrast_tau", 0.1, contrast_tau=0.1),
        make_entry("uot_rho", "0.5_1", uot_rho_a=0.5, uot_rho_b=1.0),
        make_entry("uot_rho", "1_0.5", uot_rho_a=1.0, uot_rho_b=0.5),
    ])
    return dedupe_entries(entries)


def wide_grid():
    entries = list(extended_grid())

    # Wider optimization range. The previous sweep topped out at 5e-4; keep
    # the high-LR side dense because several current best runs sit near it.
    for value in (7e-4, 1e-3):
        entries.append(make_entry("lr", value, lr=value))

    # More capacity points around the dataset defaults and a larger upper end.
    for value in (4, 12, 16, 24, 48, 64):
        entries.append(make_entry("visual_prototypes", value, num_visual_prototypes=value))

    # Sharper and smoother patch-to-prototype assignment temperatures.
    for value in (0.025, 0.075, 0.15, 0.3):
        entries.append(make_entry("proto_tau", value, proto_tau=value))

    # Wider OT smoothing and unbalanced penalty ranges.
    for value in (0.01, 0.02, 0.075, 0.15, 0.2):
        entries.append(make_entry("ot_epsilon", value, ot_epsilon=value))
    for rho_a, rho_b in ((0.1, 0.1), (0.2, 0.2), (2.0, 2.0), (0.2, 1.0), (1.0, 0.2)):
        entries.append(
            make_entry(
                "uot_rho",
                "{}_{}".format(fmt_value(rho_a), fmt_value(rho_b)),
                uot_rho_a=rho_a,
                uot_rho_b=rho_b,
            )
        )

    # Loss weights: add lower-regularization and higher-regularization points.
    entries.extend([
        make_entry("loss", "tiny_contrast", lambda_contrast=0.02, lambda_div=0.01),
        make_entry("loss", "no_contrast_keep_div", lambda_contrast=0.0, lambda_div=0.01),
        make_entry("loss", "very_strong_contrast", lambda_contrast=0.5, lambda_div=0.01),
        make_entry("loss", "very_strong_div", lambda_contrast=0.1, lambda_div=0.05),
        make_entry("loss", "regularized", lambda_contrast=0.05, lambda_div=0.02),
    ])

    # Prompt and concept-logit knobs showed dataset-dependent behavior, so add
    # nearby values instead of assuming one direction is always right.
    entries.extend([
        make_entry("prompt_ctx", 1, concept_prompt_n_ctx=1),
        make_entry("prompt_ctx", 16, concept_prompt_n_ctx=16),
        make_entry("prompt_templates", 16, concept_prompt_template_count=16),
        make_entry("prompt_templates", 22, concept_prompt_template_count=22),
        make_entry("patch_budget", 1024, max_train_patches=1024),
        make_entry("patch_budget", 12288, max_train_patches=12288),
        make_entry("patch_budget", 16384, max_train_patches=16384),
        make_entry("concept_logits", "weak", concept_logit_weight=0.25),
        make_entry("concept_logits", "very_strong", concept_logit_weight=2.0),
        make_entry("concept_logit_tau", 0.5, concept_logit_tau=0.5),
        make_entry("concept_logit_tau", 2.0, concept_logit_tau=2.0),
    ])

    # A few targeted two-parameter combinations from the current result pattern:
    # CAMELYON likes lower proto_tau/CE-only, TCGA likes prompt/proto tweaks, and
    # UBC sometimes benefits from stronger concept logits.
    entries.extend([
        make_entry("combo", "lr1e-3_tau0.05", lr=1e-3, proto_tau=0.05),
        make_entry("combo", "lr5e-4_tau0.05", lr=5e-4, proto_tau=0.05),
        make_entry("combo", "ce_tau0.05", lambda_contrast=0.0, lambda_div=0.0, proto_tau=0.05),
        make_entry("combo", "ctx8_tau0.05", concept_prompt_n_ctx=8, proto_tau=0.05),
        make_entry("combo", "proto8_tau0.05", num_visual_prototypes=8, proto_tau=0.05),
        make_entry("combo", "concept_strong_tau0.05", concept_logit_weight=1.0, proto_tau=0.05),
        make_entry("combo", "concept_strong_lr2e-4", concept_logit_weight=1.0, lr=2e-4),
    ])

    return dedupe_entries(entries)


def custom_grid(args):
    entries = []
    for spec in args.combo:
        updates = {}
        tag_parts = []
        for assignment in spec.split(","):
            key, value = assignment.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key not in DEFAULT_COMBO:
                raise ValueError("Unknown hyperparameter '{}' in --combo {}".format(key, spec))
            parsed = coerce_combo_value(key, value)
            updates[key] = parsed
            tag_parts.append("{}-{}".format(key, fmt_value(parsed)))
        if len(updates) == 1:
            axis, parsed = next(iter(updates.items()))
            entries.append(make_entry(axis, parsed, **updates))
        else:
            entries.append(make_entry("custom", "__".join(tag_parts), **updates))
    return dedupe_entries(entries)


def get_grid(args):
    if args.preset == "smoke":
        return smoke_grid()
    if args.preset == "balanced":
        return balanced_grid()
    if args.preset == "extended":
        return extended_grid()
    if args.preset == "wide":
        return wide_grid()
    if args.preset == "custom":
        if not args.combo:
            raise ValueError("--preset custom requires at least one --combo")
        return custom_grid(args)
    raise ValueError("Unknown preset {}".format(args.preset))


def has_pt_files(path):
    if not path.is_dir():
        return False
    try:
        next(path.glob("*.pt"))
        return True
    except StopIteration:
        return False


def validate_dataset(cfg, shots):
    errors = []
    if not cfg.csv_path.is_file():
        errors.append("missing csv: {}".format(cfg.csv_path))
        if cfg.key == "ubc":
            errors.append(
                "prepare UBC csv with: python tools/prepare_ubc_ocean_csv.py "
                "--metadata /path/to/UBC-OCEAN/train.csv"
            )
    if not cfg.feature_dir.is_dir():
        errors.append("missing feature dir: {}".format(cfg.feature_dir))
    elif not has_pt_files(cfg.feature_dir):
        errors.append("feature dir has no .pt files: {}".format(cfg.feature_dir))
    if not cfg.concept_bank.is_file():
        errors.append("missing concept bank: {}".format(cfg.concept_bank))
    for shot in shots:
        split_dir = ROOT / "splits" / cfg.split_template.format(shots=shot)
        if not split_dir.is_dir():
            errors.append("missing split dir: {}".format(split_dir))
    return errors


def result_csv_is_complete(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error, UnicodeDecodeError):
        return False
    if not rows:
        return False

    metric_rows = {str(row.get("metric", "")).strip().lower(): row for row in rows}
    mean_row = metric_rows.get("mean")
    if mean_row is None:
        return False
    required_any = ("val_auc", "val_f1", "val_acc", "test_auc", "test_f1", "test_acc")
    return any(mean_row.get(key) not in (None, "") for key in required_any)


def result_scope(folds, k_start, k_end):
    start = 0 if k_start in (None, -1) else int(k_start)
    end = int(folds) if k_end in (None, -1) else int(k_end)
    return int(folds), start, end


def result_filename(folds, k_start, k_end):
    total_folds, start, end = result_scope(folds, k_start, k_end)
    if end - start != total_folds:
        return "result_partial_{}_{}.csv".format(start, end - 1)
    return "result.csv"


def parse_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def result_path(job):
    existing = job.get("existing_result_csv")
    if existing and result_csv_is_complete(existing):
        return Path(existing)
    if job.get("expected_result_csv"):
        return Path(job["expected_result_csv"])
    return Path(job["results_dir"]) / "{}_s{}".format(job["exp_code"], job["seed"]) / "result.csv"


def parse_seed_from_exp_dir(name):
    match = re.search(r"_s(\d+)$", name)
    return int(match.group(1)) if match else None


def parse_shot_dir(name):
    match = re.match(r"(\d+)shots$", name)
    return int(match.group(1)) if match else None


def read_experiment_args(result_csv):
    exp_files = sorted(result_csv.parent.glob("experiment_*.txt"))
    if not exp_files:
        return {}
    try:
        value = ast.literal_eval(exp_files[0].read_text(errors="replace").strip())
    except (OSError, ValueError, SyntaxError):
        return {}
    return value if isinstance(value, dict) else {}


def combo_from_experiment(exp):
    combo = dict(DEFAULT_COMBO)
    for key in DEFAULT_COMBO:
        if key in exp and exp[key] not in (None, ""):
            try:
                combo[key] = coerce_combo_value(key, exp[key])
            except (TypeError, ValueError):
                return None
        elif key in LEGACY_MISSING_DEFAULTS:
            combo[key] = LEGACY_MISSING_DEFAULTS[key]
    return combo


def existing_result_index(results_root, run_name, folds, k_start, k_end):
    run_root = Path(results_root) / run_name
    if not run_root.is_dir():
        return {}

    target_scope = result_scope(folds, k_start, k_end)
    index = {}
    for result_csv in sorted(run_root.rglob("result*.csv")):
        if result_csv.name.startswith("summary"):
            continue
        if not result_csv_is_complete(result_csv):
            continue
        try:
            parts = result_csv.parent.relative_to(run_root).parts
        except ValueError:
            continue
        if len(parts) < 3:
            continue

        dataset = parts[0]
        shot = parse_shot_dir(parts[1])
        seed = parse_seed_from_exp_dir(result_csv.parent.name)
        if shot is None or seed is None:
            continue

        exp = read_experiment_args(result_csv)
        exp_scope = result_scope(
            parse_int(exp.get("num_splits"), folds),
            parse_int(exp.get("k_start"), -1),
            parse_int(exp.get("k_end"), -1),
        )
        if exp_scope != target_scope:
            continue

        combo = combo_from_experiment(exp)
        if combo is None:
            continue
        key = (dataset, shot, seed, target_scope, canonical_combo(combo))
        index.setdefault(key, str(result_csv))
    return index


def build_jobs(args):
    selected = parse_csv_list(args.datasets)
    if selected == ["all"] or not selected:
        selected = ["tcga", "camelyon", "ubc", "rcc"]

    shots = parse_csv_list(args.shots, int)
    seeds = parse_csv_list(args.seeds, int)
    configs = dataset_configs(args)
    missing = {}
    valid_configs = []
    for key in selected:
        if key not in configs:
            raise ValueError("Unknown dataset '{}'. Choose from {}".format(key, sorted(configs)))
        errors = validate_dataset(configs[key], shots)
        if errors:
            missing[key] = errors
        else:
            valid_configs.append(configs[key])

    if missing and not args.allow_missing_datasets:
        lines = ["Dataset validation failed:"]
        for key, errors in missing.items():
            lines.append("  [{}]".format(key))
            lines.extend("    - {}".format(error) for error in errors)
        lines.append(
            "Create missing 5-fold few-shot splits with: "
            "python tools/create_libra_fewshot_splits.py --datasets {} --shots {} --folds {} --seed 1 --overwrite".format(
                args.datasets, args.shots, args.folds
            )
        )
        lines.append("Pass missing dataset paths with --ubc-csv-path/--ubc-feature-dir or set CCA_MIL_UBC_CSV/CCA_MIL_UBC_FEATURE_DIR.")
        lines.append("For RCC, prepare metadata/splits with: python tools/prepare_tcga_rcc.py && python tools/create_libra_fewshot_splits.py --datasets rcc --shots {} --folds {} --seed 1 --overwrite".format(args.shots, args.folds))
        raise FileNotFoundError("\n".join(lines))

    grid = get_grid(args)
    existing_index = {}
    if args.skip_existing and args.skip_equivalent_existing:
        existing_index = existing_result_index(args.results_root, args.run_name, args.folds, args.k_start, args.k_end)
    scope_key = result_scope(args.folds, args.k_start, args.k_end)
    jobs = []
    for cfg in valid_configs:
        for shot in shots:
            split_dir = cfg.split_template.format(shots=shot)
            for seed in seeds:
                seen_job_keys = set()
                for entry in grid:
                    combo = apply_dataset_defaults(cfg, entry["axis"], entry["combo"])
                    canonical_key = canonical_combo(combo)
                    param_text = param_summary(combo)
                    job_key = (cfg.key, shot, seed, scope_key, canonical_key)
                    if job_key in seen_job_keys:
                        print("[dedupe-job] drop duplicate {}".format(entry["param_id"]), flush=True)
                        continue
                    seen_job_keys.add(job_key)

                    axis = entry["axis"]
                    setting = entry["setting"]
                    param_id = entry["param_id"]
                    exp_code = "{}_{}shots_{}_{}_{}".format(cfg.name, shot, args.preset, axis, setting)
                    results_dir = (
                        Path(args.results_root)
                        / args.run_name
                        / cfg.key
                        / "{}shots".format(shot)
                        / axis
                        / setting
                    )
                    log_path = (
                        Path(args.logs_root)
                        / args.run_name
                        / cfg.key
                        / "{}shots".format(shot)
                        / axis
                        / "{}_seed{}.log".format(setting, seed)
                    )

                    cmd = [
                        args.python,
                        "main.py",
                        "--seed", str(seed),
                        "--lr", str(combo["lr"]),
                        "--max_epochs", str(args.max_epochs),
                        "--k", str(args.folds),
                        "--label_frac", str(args.label_frac),
                        "--bag_loss", "ce",
                        "--task", cfg.task,
                        "--csv_path", str(cfg.csv_path),
                        "--results_dir", str(results_dir),
                        "--exp_code", exp_code,
                        "--model_type", "CCA_MIL",
                        "--mode", "transformer",
                        "--data_root_dir", args.data_root_dir,
                        "--data_folder_s", str(cfg.feature_dir),
                        "--data_folder_l", str(cfg.feature_dir),
                        "--split_dir", split_dir,
                        "--concept_bank_path", str(cfg.concept_bank),
                        "--conch_ckpt_path", args.conch_ckpt_path,
                        "--num_visual_prototypes", str(combo["num_visual_prototypes"]),
                        "--proto_tau", str(combo["proto_tau"]),
                        "--ot_epsilon", str(combo["ot_epsilon"]),
                        "--sinkhorn_iter", str(combo["sinkhorn_iter"]),
                        "--uot_rho_a", str(combo["uot_rho_a"]),
                        "--uot_rho_b", str(combo["uot_rho_b"]),
                        "--concept_pooling", str(combo["concept_pooling"]),
                        "--lambda_contrast", str(combo["lambda_contrast"]),
                        "--lambda_div", str(combo["lambda_div"]),
                        "--contrast_tau", str(combo["contrast_tau"]),
                        "--common_concept_weight", str(combo["common_concept_weight"]),
                        "--concept_prompt_n_ctx", str(combo["concept_prompt_n_ctx"]),
                        "--concept_prompt_template_count", str(combo["concept_prompt_template_count"]),
                        "--max_train_patches", str(combo["max_train_patches"]),
                        "--max_eval_patches", str(combo["max_eval_patches"]),
                        "--concept_logit_weight", str(combo["concept_logit_weight"]),
                        "--concept_logit_tau", str(combo["concept_logit_tau"]),
                    ]
                    if bool(combo["train_concept_prompt"]):
                        cmd.append("--train_concept_prompt")
                    else:
                        cmd.append("--freeze_concept_prompt")
                    if args.drop_out:
                        cmd.append("--drop_out")
                    if args.early_stopping:
                        cmd.append("--early_stopping")
                        cmd.extend(["--early_stopping_patience", str(args.early_stopping_patience)])
                        cmd.extend(["--early_stopping_stop_epoch", str(args.early_stopping_stop_epoch)])
                    if args.log_data:
                        cmd.append("--log_data")
                    if args.k_start is not None:
                        cmd.extend(["--k_start", str(args.k_start)])
                    if args.k_end is not None:
                        cmd.extend(["--k_end", str(args.k_end)])
                    if args.extra_args:
                        cmd.extend(parse_csv_list(args.extra_args))

                    existing_result_csv = existing_index.get(job_key, "")
                    expected_result_csv = (
                        results_dir
                        / "{}_s{}".format(exp_code, seed)
                        / result_filename(args.folds, args.k_start, args.k_end)
                    )
                    jobs.append({
                        "dataset": cfg.key,
                        "dataset_name": cfg.name,
                        "shot": shot,
                        "seed": seed,
                        "axis": axis,
                        "setting": setting,
                        "tag": entry["tag"],
                        "param_id": param_id,
                        "param_summary": param_text,
                        "canonical_key": canonical_key,
                        "exp_code": exp_code,
                        "results_dir": str(results_dir),
                        "log_path": str(log_path),
                        "existing_result_csv": existing_result_csv,
                        "expected_result_csv": str(expected_result_csv),
                        "cmd": cmd,
                        **combo,
                    })

    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    return jobs, missing


def write_manifest(jobs, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "jobs.csv"
    commands = run_dir / "commands.sh"
    fields = [
        "dataset", "dataset_name", "shot", "seed", "axis", "setting", "tag",
        "param_id", "param_summary", "canonical_key", "exp_code",
        "results_dir", "log_path", "existing_result_csv", "expected_result_csv", "lr", "num_visual_prototypes", "proto_tau",
        "ot_epsilon", "sinkhorn_iter", "uot_rho_a", "uot_rho_b", "concept_pooling",
        "lambda_contrast", "lambda_div", "contrast_tau", "common_concept_weight",
        "train_concept_prompt", "concept_prompt_n_ctx", "concept_prompt_template_count",
        "max_train_patches", "max_eval_patches", "concept_logit_weight", "concept_logit_tau",
    ]
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            writer.writerow({key: job.get(key) for key in fields})
    with commands.open("w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\ncd '{}'\n\n".format(ROOT))
        for job in jobs:
            f.write(" ".join(shell_quote(part) for part in job["cmd"]))
            f.write("\n")
    commands.chmod(0o755)
    return manifest, commands


def shell_quote(value):
    value = str(value)
    if all(ch.isalnum() or ch in "._-/:=," for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def read_metric_csv(path):
    if not path.is_file():
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    mean_row = next((row for row in rows if row.get("metric") == "mean"), None)
    std_row = next((row for row in rows if row.get("metric") == "var"), None)
    if mean_row is None:
        return None
    metrics = {}
    for key, value in mean_row.items():
        if key == "metric" or value in (None, ""):
            continue
        try:
            metrics[key + "_mean"] = float(value)
        except ValueError:
            pass
    if std_row is not None:
        for key, value in std_row.items():
            if key == "metric" or value in (None, ""):
                continue
            try:
                metrics[key + "_std"] = float(value)
            except ValueError:
                pass
    return metrics


def infer_missing_status(job):
    log_path = Path(job["log_path"])
    if not log_path.is_file():
        return "missing_not_started"
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return "missing"
    # Logs are appended across retries. Only inspect the latest attempt so an
    # old OOM does not make an in-progress or retried job look failed forever.
    marker = "===== START "
    last_start = text.rfind(marker)
    if last_start >= 0:
        text = text[last_start:]
    lowered = text.lower()
    if "outofmemoryerror" in lowered or "out of memory" in lowered or "cuda error: out of memory" in lowered:
        return "failed_oom"
    if "traceback (most recent call last)" in lowered:
        return "failed_traceback"
    if "===== start " in lowered:
        return "missing_after_start"
    return "missing"


def collect_results(jobs, run_dir, rank_metric):
    rows = []
    for job in jobs:
        metrics = read_metric_csv(result_path(job))
        status = "done" if metrics else infer_missing_status(job)
        row = {
            "status": status,
            "dataset": job["dataset"],
            "shot": job["shot"],
            "seed": job["seed"],
            "axis": job["axis"],
            "setting": job["setting"],
            "tag": job["tag"],
            "param_id": job["param_id"],
            "param_summary": job["param_summary"],
            "canonical_key": job["canonical_key"],
            "result_csv": str(result_path(job)),
            "log_path": job["log_path"],
            "existing_result_csv": job.get("existing_result_csv", ""),
        }
        row.update({key: job[key] for key in (
            "lr", "num_visual_prototypes", "proto_tau", "ot_epsilon", "sinkhorn_iter",
            "uot_rho_a", "uot_rho_b", "concept_pooling", "lambda_contrast",
            "lambda_div", "contrast_tau", "common_concept_weight",
            "train_concept_prompt", "concept_prompt_n_ctx", "concept_prompt_template_count",
            "max_train_patches", "max_eval_patches", "concept_logit_weight", "concept_logit_tau",
        )})
        if metrics:
            row.update(metrics)
        rows.append(row)

    def sort_key(row):
        value = row.get(rank_metric)
        if value is None:
            return -1.0
        return value

    rows.sort(key=sort_key, reverse=True)
    out_path = run_dir / "sweep_summary.csv"
    all_fields = []
    for row in rows:
        for key in row:
            if key not in all_fields:
                all_fields.append(key)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def stop_running_jobs(running, timeout=15):
    for gpu, items in running.items():
        for item in items:
            proc = item["proc"]
            job = item["job"]
            if proc.poll() is not None:
                continue
            print("[interrupt] terminate gpu={} pid={} {}".format(gpu, proc.pid, job["exp_code"]), flush=True)
            proc.terminate()

    deadline = time.time() + timeout
    for gpu, items in running.items():
        for item in items:
            proc = item["proc"]
            remaining = max(0.0, deadline - time.time())
            if proc.poll() is not None:
                continue
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print("[interrupt] kill gpu={} pid={} {}".format(gpu, proc.pid, item["job"]["exp_code"]), flush=True)
                proc.kill()

    for items in running.values():
        for item in items:
            try:
                item["log_file"].close()
            except OSError:
                pass


def run_jobs(jobs, args, run_dir):
    gpus = parse_csv_list(args.gpus)
    if not gpus:
        raise ValueError("--gpus must not be empty")
    per_gpu = max(1, args.max_jobs_per_gpu)
    pending = list(jobs)
    running = {gpu: [] for gpu in gpus}
    finished = 0
    failed = 0
    skipped = 0

    try:
        while pending or any(running.values()):
            for gpu in gpus:
                survivors = []
                for item in running[gpu]:
                    proc = item["proc"]
                    code = proc.poll()
                    if code is None:
                        survivors.append(item)
                        continue
                    item["log_file"].close()
                    finished += 1
                    if code != 0:
                        failed += 1
                        print("[fail] gpu={} code={} {}".format(gpu, code, item["job"]["exp_code"]), flush=True)
                    else:
                        print("[done] gpu={} {}".format(gpu, item["job"]["exp_code"]), flush=True)
                running[gpu] = survivors

            for gpu in gpus:
                while pending and len(running[gpu]) < per_gpu:
                    job = pending.pop(0)
                    if args.skip_existing and result_csv_is_complete(result_path(job)):
                        skipped += 1
                        print("[skip] {} -> {}".format(job["exp_code"], result_path(job)), flush=True)
                        continue

                    Path(job["results_dir"]).mkdir(parents=True, exist_ok=True)
                    log_path = Path(job["log_path"])
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_file = log_path.open("a")
                    log_file.write("\n\n===== START {} GPU {} =====\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), gpu))
                    log_file.write(" ".join(shell_quote(part) for part in job["cmd"]) + "\n")
                    log_file.flush()

                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    env.setdefault("PYTHONUNBUFFERED", "1")
                    env.setdefault("OMP_NUM_THREADS", "1")
                    env.setdefault("MKL_NUM_THREADS", "1")
                    env.setdefault("OPENBLAS_NUM_THREADS", "1")
                    env.setdefault("NUMEXPR_NUM_THREADS", "1")
                    proc = subprocess.Popen(
                        job["cmd"],
                        cwd=str(ROOT),
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                    )
                    running[gpu].append({"proc": proc, "job": job, "log_file": log_file})
                    print("[start] gpu={} pid={} {}".format(gpu, proc.pid, job["exp_code"]), flush=True)

            collect_results(jobs, run_dir, args.rank_metric)
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\n[interrupt] stopping scheduler and running child jobs...", flush=True)
        stop_running_jobs(running)
        summary_path = collect_results(jobs, run_dir, args.rank_metric)
        print("[interrupt] partial summary: {}".format(summary_path), flush=True)
        raise

    summary_path = collect_results(jobs, run_dir, args.rank_metric)
    return finished, failed, skipped, summary_path


def parse_args():
    parser = argparse.ArgumentParser(description="AOT-MIL hyperparameter sweep scheduler")
    parser.add_argument("--datasets", default="all", help="Comma list: all,tcga,camelyon,ubc,rcc")
    parser.add_argument("--shots", default="1,4,16", help="Comma list, e.g. 1,4,16")
    parser.add_argument("--seeds", default="1", help="Comma list of random seeds")
    parser.add_argument("--preset", choices=["smoke", "balanced", "extended", "wide", "custom"], default="balanced")
    parser.add_argument("--combo", action="append", default=[], help="Custom combo like num_visual_prototypes=16,proto_tau=0.1")
    parser.add_argument("--run-name", default=time.strftime("aot_sweep_%Y%m%d_%H%M%S"))
    parser.add_argument("--results-root", default=str(DEFAULT_OUTPUT_ROOT / "results" / "AOT_MIL_sweeps"))
    parser.add_argument("--logs-root", default=str(DEFAULT_OUTPUT_ROOT / "logs" / "AOT_MIL_sweeps"))
    parser.add_argument("--gpus", default="0", help="Comma list of physical GPU ids")
    parser.add_argument("--max-jobs-per-gpu", type=int, default=1)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", dest="max_epochs", type=int, default=80)
    parser.add_argument("--k-start", type=int, default=None)
    parser.add_argument("--k-end", type=int, default=None)
    parser.add_argument("--label-frac", type=float, default=1.0)
    parser.add_argument("--data-root-dir", default="/data2/yuhaowang/WSIFew")
    parser.add_argument("--conch-ckpt-path", default=str(ROOT / "ckg" / "pytorch_model.bin"))
    parser.add_argument("--tcga-csv-path", default=None)
    parser.add_argument("--camelyon-csv-path", default=None)
    parser.add_argument("--ubc-csv-path", default=None)
    parser.add_argument("--rcc-csv-path", default=None)
    parser.add_argument("--tcga-feature-dir", default=None)
    parser.add_argument("--camelyon-feature-dir", default=None)
    parser.add_argument("--ubc-feature-dir", default=None)
    parser.add_argument("--rcc-feature-dir", default=None)
    parser.add_argument("--allow-missing-datasets", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--skip-equivalent-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-equivalent-existing", dest="skip_equivalent_existing", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--rank-metric", default="val_auc_mean")
    parser.add_argument("--early-stopping", action="store_true", default=True)
    parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false")
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--early-stopping-stop-epoch", type=int, default=0)
    parser.add_argument("--drop-out", action="store_true", default=True)
    parser.add_argument("--no-drop-out", dest="drop_out", action="store_false")
    parser.add_argument("--log-data", action="store_true", default=False)
    parser.add_argument("--extra-args", default=None, help="Comma-separated extra argv tokens appended to main.py")
    return parser.parse_args()


def main():
    args = parse_args()
    jobs, missing = build_jobs(args)
    run_dir = Path(args.logs_root) / args.run_name
    manifest, commands = write_manifest(jobs, run_dir)
    completed_jobs = sum(1 for job in jobs if result_csv_is_complete(result_path(job)))
    print("Prepared {} jobs".format(len(jobs)))
    print("Already completed/equivalent jobs: {}; pending jobs: {}".format(completed_jobs, len(jobs) - completed_jobs))
    print("Manifest: {}".format(manifest))
    print("Commands: {}".format(commands))
    if missing and args.allow_missing_datasets:
        print("Skipped missing datasets: {}".format(", ".join(sorted(missing))))

    if args.validate_only or args.dry_run:
        summary_path = collect_results(jobs, run_dir, args.rank_metric)
        print("Dry/validate run only. Summary placeholder: {}".format(summary_path))
        return 0

    if args.collect_only:
        summary_path = collect_results(jobs, run_dir, args.rank_metric)
        print("Collected results: {}".format(summary_path))
        return 0

    try:
        finished, failed, skipped, summary_path = run_jobs(jobs, args, run_dir)
    except KeyboardInterrupt:
        return 130
    print("Finished: {}, failed: {}, skipped: {}".format(finished, failed, skipped))
    print("Sweep summary: {}".format(summary_path))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
