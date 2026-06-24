#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${RAW_DIR:-/data2/yuhaowang/WSIFew/TCGA-NSCLC}"
PROCESSED_DIR="${PROCESSED_DIR:-/data2/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC}"
DATASET_CSV="${DATASET_CSV:-$ROOT_DIR/dataset_csv/LUAD_LUSC.csv}"
CONCH_CKPT="${CONCH_CKPT:-$ROOT_DIR/ckg/pytorch_model.bin}"
GPUS="${GPUS:-2,3,4,5,6,7}"
PATCH_WORKERS="${PATCH_WORKERS:-8}"
CONTOUR_WORKERS="${CONTOUR_WORKERS:-4}"
FEATURE_WORKERS="${FEATURE_WORKERS:-6}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PATCH_LIMIT="${PATCH_LIMIT:-0}"
FEATURE_LIMIT="${FEATURE_LIMIT:-0}"
SKIP_AUDIT="${SKIP_AUDIT:-0}"

PATCH_LIMIT_ARGS=()
if [[ "$PATCH_LIMIT" != "0" ]]; then
  PATCH_LIMIT_ARGS=(--limit_slides "$PATCH_LIMIT")
fi

FEATURE_LIMIT_ARGS=()
if [[ "$FEATURE_LIMIT" != "0" ]]; then
  FEATURE_LIMIT_ARGS=(--limit_slides "$FEATURE_LIMIT")
fi

cd "$ROOT_DIR"
mkdir -p logs

echo "[1/3] Patch extraction resume"
conda run -n hest python fast_create_patches_fp.py \
  --source "$RAW_DIR" \
  --save_dir "$PROCESSED_DIR" \
  --slide_exts .svs \
  --preset tcga.csv \
  --refresh_pending_params \
  --patch --seg --stitch \
  --patch_size 512 \
  --step_size 512 \
  --patch_level 0 \
  --num_workers "$PATCH_WORKERS" \
  --contour_workers "$CONTOUR_WORKERS" \
  "${PATCH_LIMIT_ARGS[@]}"

echo "[2/3] CONCH feature extraction resume"
conda run -n hest python fast_extract_features_fp.py \
  --data_h5_dir "$PROCESSED_DIR" \
  --data_slide_dir "$RAW_DIR" \
  --csv_path "$PROCESSED_DIR/process_list_autogen.csv" \
  --feat_dir "$PROCESSED_DIR/feature" \
  --slide_ext .svs \
  --model_name conch_v1 \
  --conch_ckpt_path "$CONCH_CKPT" \
  --target_patch_size 448 \
  --batch_size "$BATCH_SIZE" \
  --gpus "$GPUS" \
  --num_workers "$FEATURE_WORKERS" \
  --progress_interval 1 \
  "${FEATURE_LIMIT_ARGS[@]}"

echo "[3/3] Audit"
if [[ "$SKIP_AUDIT" == "1" ]]; then
  echo "Skipping audit because SKIP_AUDIT=1"
  exit 0
fi

conda run -n hest python tools/audit_preprocessing.py \
  --raw_dir "$RAW_DIR" \
  --processed_dir "$PROCESSED_DIR" \
  --feat_dir "$PROCESSED_DIR/feature" \
  --dataset_csv "$DATASET_CSV" \
  --split_dir splits/LUAD_LUSC_1shots_5folds splits/LUAD_LUSC_4shots_5folds splits/LUAD_LUSC_16shots_5folds \
  --slide_exts .svs \
  --strict

echo "TCGA-NSCLC preprocessing is complete."
