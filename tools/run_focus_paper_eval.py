#!/usr/bin/env python3
# coding=utf-8
"""Run FOCUS paper-style 10-fold few-shot evaluation.

This scheduler is intentionally separate from the CCA/AOT-MIL sweep scripts.
It runs FOCUS with the legacy 10-fold splits for 4/8/16-shot experiments and
stores each dataset-shot result in an isolated directory. Use --fold-jobs to
schedule every fold as an independent process for high-throughput multi-GPU
runs.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("/data2/yuhaowang/cca-mil-result")
DEFAULT_PYTHON = os.environ.get("CCA_MIL_PYTHON", sys.executable)


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    name: str
    task: str
    csv_path: Path
    feature_dir: Path
    prompt_path: Path
    split_template: str


def parse_csv_list(value, cast=str):
    if value is None or value == "":
        return []
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


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

    return {
        "tcga": DatasetConfig(
            key="tcga",
            name="LUAD_LUSC",
            task="task_tcga_lung_subtyping",
            csv_path=Path(args.tcga_csv_path or ROOT / "dataset_csv" / "LUAD_LUSC.csv"),
            feature_dir=tcga_feature,
            prompt_path=Path(args.tcga_prompt_path or ROOT / "text_prompt" / "TCGA_Lung_two_scale_text_prompt.csv"),
            split_template="LUAD_LUSC_{shots}shots_10folds",
        ),
        "camelyon": DatasetConfig(
            key="camelyon",
            name="camelyon",
            task="task_camelyon_subtyping",
            csv_path=Path(args.camelyon_csv_path or ROOT / "dataset_csv" / "camelyon.csv"),
            feature_dir=camelyon_feature,
            prompt_path=Path(args.camelyon_prompt_path or ROOT / "text_prompt" / "CAMELYON_two_scale_text_prompt.csv"),
            split_template="camelyon_{shots}shots_10folds",
        ),
        "ubc": DatasetConfig(
            key="ubc",
            name="UBC-OCEAN",
            task="task_UBC-OCEAN_subtyping",
            csv_path=ubc_csv,
            feature_dir=ubc_feature,
            prompt_path=Path(args.ubc_prompt_path or ROOT / "text_prompt" / "UBC-OCEAN_two_scale_text_prompt.csv"),
            split_template="UBC-OCEAN_{shots}shots_10folds",
        ),
    }


def shell_quote(value):
    value = str(value)
    if all(ch.isalnum() or ch in "._-/:=," for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def has_pt_files(path):
    if not path.is_dir():
        return False
    try:
        next(path.glob("*.pt"))
        return True
    except StopIteration:
        return False


def validate_dataset(cfg, shots, folds):
    errors = []
    if not cfg.csv_path.is_file():
        errors.append("missing csv: {}".format(cfg.csv_path))
    if not cfg.prompt_path.is_file():
        errors.append("missing FOCUS prompt csv: {}".format(cfg.prompt_path))
    if not cfg.feature_dir.is_dir():
        errors.append("missing feature dir: {}".format(cfg.feature_dir))
    elif not has_pt_files(cfg.feature_dir):
        errors.append("feature dir has no .pt files: {}".format(cfg.feature_dir))
    for shot in shots:
        split_dir = ROOT / "splits" / cfg.split_template.format(shots=shot)
        if not split_dir.is_dir():
            errors.append("missing split dir: {}".format(split_dir))
            continue
        for fold in range(folds):
            split_file = split_dir / "splits_{}.csv".format(fold)
            if not split_file.is_file():
                errors.append("missing split file: {}".format(split_file))
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
    mean_row = next((row for row in rows if str(row.get("metric", "")).strip().lower() == "mean"), None)
    if mean_row is None:
        return False
    required_any = ("val_auc", "val_f1", "val_acc", "test_auc", "test_f1", "test_acc")
    return any(mean_row.get(key) not in (None, "") for key in required_any)


def result_path(job):
    if job.get("fold") not in (None, ""):
        fold = int(job["fold"])
        return Path(job["results_dir"]) / "{}_s{}".format(job["exp_code"], job["seed"]) / "result_partial_{}_{}.csv".format(fold, fold)
    return Path(job["results_dir"]) / "{}_s{}".format(job["exp_code"], job["seed"]) / "result.csv"


def read_metric_csv(path):
    path = Path(path)
    if not result_csv_is_complete(path):
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    mean_row = next((row for row in rows if row.get("metric") == "mean"), None)
    std_row = next((row for row in rows if row.get("metric") in {"std", "var"}), None)
    metrics = {}
    for suffix, row in (("mean", mean_row), ("std", std_row)):
        if row is None:
            continue
        for key, value in row.items():
            if key == "metric" or value in (None, ""):
                continue
            try:
                metrics["{}_{}".format(key, suffix)] = float(value)
            except ValueError:
                metrics["{}_{}".format(key, suffix)] = value
    return metrics


def build_jobs(args):
    selected = parse_csv_list(args.datasets)
    if selected == ["all"] or not selected:
        selected = ["tcga", "camelyon", "ubc"]

    shots = parse_csv_list(args.shots, int)
    seeds = parse_csv_list(args.seeds, int)
    configs = dataset_configs(args)
    missing = {}
    valid_configs = []
    for key in selected:
        if key not in configs:
            raise ValueError("Unknown dataset '{}'. Choose from {}".format(key, sorted(configs)))
        errors = validate_dataset(configs[key], shots, args.folds)
        if errors:
            missing[key] = errors
        else:
            valid_configs.append(configs[key])

    if missing and not args.allow_missing_datasets:
        lines = ["Dataset validation failed:"]
        for key, errors in missing.items():
            lines.append("  [{}]".format(key))
            lines.extend("    - {}".format(error) for error in errors)
        raise FileNotFoundError("\n".join(lines))

    jobs = []
    for cfg in valid_configs:
        for shot in shots:
            split_dir = cfg.split_template.format(shots=shot)
            for seed in seeds:
                group_exp_code = "{}_{}shots_focus_paper_{}folds".format(cfg.name, shot, args.folds)
                results_dir = Path(args.results_root) / args.run_name / cfg.key / "{}shots".format(shot)
                fold_values = range(args.folds) if args.fold_jobs else [None]
                for fold in fold_values:
                    exp_code = "{}_fold{}".format(group_exp_code, fold) if fold is not None else group_exp_code
                    log_name = "seed{}_fold{}.log".format(seed, fold) if fold is not None else "seed{}.log".format(seed)
                    log_path = Path(args.logs_root) / args.run_name / cfg.key / "{}shots".format(shot) / log_name
                    cmd = [
                        args.python,
                        "main.py",
                        "--seed", str(seed),
                        "--lr", str(args.lr),
                        "--max_epochs", str(args.max_epochs),
                        "--k", str(args.folds),
                        "--label_frac", str(args.label_frac),
                        "--bag_loss", args.bag_loss,
                        "--task", cfg.task,
                        "--csv_path", str(cfg.csv_path),
                        "--results_dir", str(results_dir),
                        "--exp_code", exp_code,
                        "--model_type", "FOCUS",
                        "--mode", "transformer",
                        "--data_root_dir", args.data_root_dir,
                        "--data_folder_s", str(cfg.feature_dir),
                        "--data_folder_l", str(cfg.feature_dir),
                        "--split_dir", split_dir,
                        "--text_prompt_path", str(cfg.prompt_path),
                        "--conch_ckpt_path", args.conch_ckpt_path,
                        "--max_context_length", str(args.max_context_length),
                        "--window_size", str(args.window_size),
                        "--sim_threshold", str(args.sim_threshold),
                        "--prototype_number", str(args.prototype_number),
                    ]
                    if args.fold_jobs:
                        cmd.extend(["--k_start", str(fold), "--k_end", str(fold + 1)])
                    elif args.k_start is not None:
                        cmd.extend(["--k_start", str(args.k_start)])
                    if not args.fold_jobs and args.k_end is not None:
                        cmd.extend(["--k_end", str(args.k_end)])
                    if args.drop_out:
                        cmd.append("--drop_out")
                    if args.early_stopping:
                        cmd.append("--early_stopping")
                        cmd.extend(["--early_stopping_patience", str(args.early_stopping_patience)])
                        cmd.extend(["--early_stopping_stop_epoch", str(args.early_stopping_stop_epoch)])
                    if args.log_data:
                        cmd.append("--log_data")
                    if args.extra_args:
                        cmd.extend(parse_csv_list(args.extra_args))

                    jobs.append({
                        "dataset": cfg.key,
                        "dataset_name": cfg.name,
                        "shot": shot,
                        "seed": seed,
                        "fold": "" if fold is None else fold,
                        "exp_code": exp_code,
                        "group_exp_code": group_exp_code,
                        "results_dir": str(results_dir),
                        "log_path": str(log_path),
                        "split_dir": split_dir,
                        "csv_path": str(cfg.csv_path),
                        "feature_dir": str(cfg.feature_dir),
                        "prompt_path": str(cfg.prompt_path),
                        "cmd": cmd,
                    })

    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    return jobs, missing


def write_manifest(jobs, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "jobs.csv"
    commands = run_dir / "commands.sh"
    fields = [
        "dataset", "dataset_name", "shot", "seed", "fold", "exp_code", "group_exp_code", "split_dir",
        "csv_path", "feature_dir", "prompt_path", "results_dir", "log_path", "result_csv",
    ]
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            row = {key: job.get(key, "") for key in fields}
            row["result_csv"] = str(result_path(job))
            writer.writerow(row)
    with commands.open("w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\ncd '{}'\n\n".format(ROOT))
        for job in jobs:
            f.write(" ".join(shell_quote(part) for part in job["cmd"]))
            f.write("\n")
    commands.chmod(0o755)
    return manifest, commands


def collect_results(jobs, run_dir, rank_metric):
    fold_rows = []
    for job in jobs:
        metrics = read_metric_csv(result_path(job))
        row = {
            "status": "done" if metrics else "missing",
            "dataset": job["dataset"],
            "dataset_name": job["dataset_name"],
            "shot": job["shot"],
            "seed": job["seed"],
            "fold": job.get("fold", ""),
            "exp_code": job["exp_code"],
            "group_exp_code": job.get("group_exp_code", job["exp_code"]),
            "split_dir": job["split_dir"],
            "results_dir": job["results_dir"],
            "result_csv": str(result_path(job)),
            "log_path": job["log_path"],
        }
        if metrics:
            row.update(metrics)
        fold_rows.append(row)

    group_rows = []
    grouped = {}
    for row in fold_rows:
        key = (row["dataset"], row["dataset_name"], row["shot"], row["seed"], row["group_exp_code"], row["split_dir"], row["results_dir"])
        grouped.setdefault(key, []).append(row)

    metric_names = ("val_auc", "val_f1", "val_acc", "test_auc", "test_f1", "test_acc")
    for key, rows in grouped.items():
        dataset, dataset_name, shot, seed, exp_code, split_dir, results_dir = key
        done_rows = [row for row in rows if row["status"] == "done"]
        group = {
            "status": "done" if len(done_rows) == len(rows) else "partial" if done_rows else "missing",
            "dataset": dataset,
            "dataset_name": dataset_name,
            "shot": shot,
            "seed": seed,
            "num_folds_done": len(done_rows),
            "num_folds_total": len(rows),
            "exp_code": exp_code,
            "split_dir": split_dir,
            "results_dir": results_dir,
        }
        for metric in metric_names:
            values = []
            for row in done_rows:
                value = row.get("{}_mean".format(metric))
                if isinstance(value, (int, float)):
                    values.append(float(value))
            if values:
                group["{}_mean".format(metric)] = sum(values) / len(values)
                if len(values) > 1:
                    mean = group["{}_mean".format(metric)]
                    group["{}_std".format(metric)] = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
                else:
                    group["{}_std".format(metric)] = 0.0
        group_rows.append(group)

    def sort_key(row):
        value = row.get(rank_metric)
        if value is None:
            return -1.0
        return value

    group_rows.sort(key=sort_key, reverse=True)
    fold_rows.sort(key=lambda row: (row["dataset"], int(row["shot"]), int(row["seed"]), str(row.get("fold", ""))))

    fold_out_path = run_dir / "focus_paper_fold_results.csv"
    fold_fields = []
    for row in fold_rows:
        for key in row:
            if key not in fold_fields:
                fold_fields.append(key)
    with fold_out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fold_fields)
        writer.writeheader()
        writer.writerows(fold_rows)

    out_path = run_dir / "focus_paper_summary.csv"
    fields = []
    for row in group_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(group_rows)
    return out_path


PROGRESS_MARKERS = (
    "Epoch:",
    "Val Set",
    "Val error:",
    "Test error:",
    "Saving model",
    "Early stopping",
    "Traceback",
    "RuntimeError",
    "CUDA out",
    "Killed",
)


def format_seconds(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return "{}s".format(seconds)
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return "{}m{}s".format(minutes, seconds)
    hours, minutes = divmod(minutes, 60)
    return "{}h{}m".format(hours, minutes)


def latest_log_progress(log_path, max_scan_lines=240, max_items=2):
    path = Path(log_path)
    if not path.is_file():
        return "log not created"

    matches = []
    try:
        with path.open(errors="replace") as f:
            for line in deque(f, maxlen=max_scan_lines):
                text = line.strip()
                if text and any(marker in text for marker in PROGRESS_MARKERS):
                    matches.append(text)
    except OSError as exc:
        return "log read failed: {}".format(exc)

    if matches:
        return " | ".join(matches[-max_items:])

    try:
        with path.open(errors="replace") as f:
            tail = [line.strip() for line in deque(f, maxlen=1)]
        return tail[0] if tail and tail[0] else "log has no progress line yet"
    except OSError as exc:
        return "log read failed: {}".format(exc)


def log_age(log_path):
    path = Path(log_path)
    if not path.is_file():
        return "n/a"
    try:
        return format_seconds(time.time() - path.stat().st_mtime)
    except OSError:
        return "n/a"


def print_live_status(running, pending_count, finished, failed, skipped, args):
    if args.no_live_status:
        return

    running_items = []
    for gpu, items in running.items():
        for item in items:
            running_items.append((gpu, item))

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        "[status] {} running={} pending={} finished={} failed={} skipped={}".format(
            timestamp, len(running_items), pending_count, finished, failed, skipped
        ),
        flush=True,
    )
    for gpu, item in running_items:
        job = item["job"]
        proc = item["proc"]
        log_path = Path(job["log_path"])
        result_state = "done" if result_csv_is_complete(result_path(job)) else "pending"
        progress = latest_log_progress(log_path, max_items=args.status_lines)
        print(
            "[running] gpu={} pid={} {} result={} log_age={} log={} :: {}".format(
                gpu,
                proc.pid,
                job["exp_code"],
                result_state,
                log_age(log_path),
                log_path,
                progress,
            ),
            flush=True,
        )


def find_running_pids_by_exp_code(jobs):
    exp_codes = [job["exp_code"] for job in jobs]
    pids = {exp_code: [] for exp_code in exp_codes}
    try:
        output = subprocess.check_output(["ps", "-eo", "pid=,comm=,args="], text=True, errors="replace")
    except (OSError, subprocess.CalledProcessError):
        return pids

    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 2)
        if len(parts) < 3:
            continue
        pid, command_name, command_line = parts
        if not command_name.startswith("python"):
            continue
        if " main.py " not in " {} ".format(command_line):
            continue
        for exp_code in exp_codes:
            if exp_code in command_line:
                pids[exp_code].append(pid)
                break
    return pids


def print_monitor_snapshot(jobs, args):
    pids = find_running_pids_by_exp_code(jobs)
    done = 0
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print("[monitor] {}".format(timestamp), flush=True)
    for job in jobs:
        result_done = result_csv_is_complete(result_path(job))
        done += int(result_done)
        pid_text = ",".join(pids.get(job["exp_code"], [])) or "-"
        log_path = Path(job["log_path"])
        progress = latest_log_progress(log_path, max_items=args.status_lines)
        print(
            "[job] pid={} {} result={} log_age={} log={} :: {}".format(
                pid_text,
                job["exp_code"],
                "done" if result_done else "pending",
                log_age(log_path),
                log_path,
                progress,
            ),
            flush=True,
        )
    print("[monitor] done={}/{}".format(done, len(jobs)), flush=True)
    return done


def stop_running_jobs(running, timeout=15):
    for gpu, items in running.items():
        for item in items:
            proc = item["proc"]
            if proc.poll() is None:
                print("[interrupt] terminate gpu={} pid={} {}".format(gpu, proc.pid, item["job"]["exp_code"]), flush=True)
                proc.terminate()

    deadline = time.time() + timeout
    for gpu, items in running.items():
        for item in items:
            proc = item["proc"]
            if proc.poll() is not None:
                continue
            try:
                proc.wait(timeout=max(0.0, deadline - time.time()))
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

            while pending and any(len(running[gpu]) < per_gpu for gpu in gpus):
                launched_or_skipped = False
                for gpu in gpus:
                    if not pending:
                        break
                    if len(running[gpu]) >= per_gpu:
                        continue

                    job = pending.pop(0)
                    launched_or_skipped = True
                    if args.skip_existing and result_csv_is_complete(result_path(job)):
                        skipped += 1
                        print("[skip] {} -> {}".format(job["exp_code"], result_path(job)), flush=True)
                        continue
                    if not args.allow_duplicate_running:
                        existing_pids = find_running_pids_by_exp_code([job]).get(job["exp_code"], [])
                        if existing_pids:
                            skipped += 1
                            print(
                                "[skip-running] {} already running pid={}".format(
                                    job["exp_code"], ",".join(existing_pids)
                                ),
                                flush=True,
                            )
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
                    env["PYTHONUNBUFFERED"] = "1"
                    proc = subprocess.Popen(
                        job["cmd"],
                        cwd=str(ROOT),
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                    )
                    running[gpu].append({"proc": proc, "job": job, "log_file": log_file})
                    print("[start] gpu={} pid={} {} log={}".format(gpu, proc.pid, job["exp_code"], log_path), flush=True)

                if not launched_or_skipped:
                    break

            collect_results(jobs, run_dir, args.rank_metric)
            print_live_status(running, len(pending), finished, failed, skipped, args)
            if not pending and not any(running.values()):
                break
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
    parser = argparse.ArgumentParser(description="FOCUS paper-style 10-fold few-shot evaluator")
    parser.add_argument("--datasets", default="all", help="Comma list: all,tcga,camelyon,ubc")
    parser.add_argument("--shots", default="4,8,16")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--run-name", default="focus_paper_10fold_4_8_16")
    parser.add_argument("--results-root", default=str(DEFAULT_OUTPUT_ROOT / "results" / "FOCUS_paper_eval"))
    parser.add_argument("--logs-root", default=str(DEFAULT_OUTPUT_ROOT / "logs" / "FOCUS_paper_eval"))
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--max-jobs-per-gpu", type=int, default=1)
    parser.add_argument("--fold-jobs", action="store_true",
                        help="Run each fold as an independent process and aggregate partial fold results")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--no-live-status", action="store_true")
    parser.add_argument("--status-lines", type=int, default=2)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--max-epochs", dest="max_epochs", type=int, default=200)
    parser.add_argument("--k-start", type=int, default=None)
    parser.add_argument("--k-end", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--label-frac", type=float, default=1.0)
    parser.add_argument("--bag-loss", default="ce")
    parser.add_argument("--data-root-dir", default="/data2/yuhaowang/WSIFew")
    parser.add_argument("--conch-ckpt-path", default=str(ROOT / "ckg" / "pytorch_model.bin"))
    parser.add_argument("--max-context-length", type=int, default=8192)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--sim-threshold", type=float, default=0.8)
    parser.add_argument("--prototype-number", type=int, default=16)
    parser.add_argument("--tcga-csv-path", default=None)
    parser.add_argument("--camelyon-csv-path", default=None)
    parser.add_argument("--ubc-csv-path", default=None)
    parser.add_argument("--tcga-feature-dir", default=None)
    parser.add_argument("--camelyon-feature-dir", default=None)
    parser.add_argument("--ubc-feature-dir", default=None)
    parser.add_argument("--tcga-prompt-path", default=None)
    parser.add_argument("--camelyon-prompt-path", default=None)
    parser.add_argument("--ubc-prompt-path", default=None)
    parser.add_argument("--early-stopping", action="store_true", default=False)
    parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false")
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-stop-epoch", type=int, default=40)
    parser.add_argument("--drop-out", action="store_true", default=True)
    parser.add_argument("--no-drop-out", dest="drop_out", action="store_false")
    parser.add_argument("--log-data", action="store_true", default=False)
    parser.add_argument("--allow-missing-datasets", action="store_true")
    parser.add_argument("--allow-duplicate-running", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--monitor-only", action="store_true")
    parser.add_argument("--monitor-once", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--rank-metric", default="val_auc_mean")
    parser.add_argument("--extra-args", default=None, help="Comma-separated extra argv tokens appended to main.py")
    return parser.parse_args()


def main():
    args = parse_args()
    jobs, missing = build_jobs(args)
    run_dir = Path(args.logs_root) / args.run_name
    manifest, commands = write_manifest(jobs, run_dir)
    print("Prepared {} FOCUS jobs".format(len(jobs)))
    print("Manifest: {}".format(manifest))
    print("Commands: {}".format(commands))
    if missing and args.allow_missing_datasets:
        print("Skipped missing datasets: {}".format(", ".join(sorted(missing))))

    if args.validate_only or args.dry_run:
        summary_path = collect_results(jobs, run_dir, args.rank_metric)
        print("Dry/validate run only. Summary placeholder: {}".format(summary_path))
        return 0

    if args.monitor_only:
        try:
            while True:
                collect_results(jobs, run_dir, args.rank_metric)
                done = print_monitor_snapshot(jobs, args)
                if args.monitor_once or done == len(jobs):
                    break
                time.sleep(args.poll_interval)
        except KeyboardInterrupt:
            return 130
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
    print("FOCUS paper summary: {}".format(summary_path))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
