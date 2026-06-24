#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/yuhaowang/cca-mil-result}"
RESULTS_ROOT="${RESULTS_ROOT:-$OUTPUT_ROOT/results}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs/AOT_MIL_sweeps}"

KNOWN_HEST_PYTHON="/home/yuhaowang/anaconda3/envs/hest/bin/python"
KNOWN_BASE_PYTHON="/home/yuhaowang/anaconda3/bin/python"
if [[ -n "${CCA_MIL_PYTHON:-}" ]]; then
  PYTHON_BIN="$CCA_MIL_PYTHON"
elif [[ -n "${PYTHON_BIN:-}" && "$PYTHON_BIN" != "$KNOWN_BASE_PYTHON" ]]; then
  :
elif [[ -n "${CONDA_PREFIX:-}" && "$CONDA_PREFIX/bin/python" != "$KNOWN_BASE_PYTHON" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [[ -x "$KNOWN_HEST_PYTHON" ]]; then
  PYTHON_BIN="$KNOWN_HEST_PYTHON"
else
  PYTHON_BIN="$(command -v python)"
fi

RUN_NAME="${RUN_NAME:-cca_mil_ablation_best}"
DATASETS="${DATASETS:-rcc}"
SHOTS="${SHOTS:-1,4,16}"
PRESET="${PRESET:-wide}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-3}"
FOLDS="${FOLDS:-5}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
SEEDS="${SEEDS:-1}"
RANK_METRIC="${RANK_METRIC:-val_auc_mean}"
LIBRA_SUMMARY="${LIBRA_SUMMARY:-$RESULTS_ROOT/Libra-MIL/summary_results.csv}"
CHECK_ONLY="${CHECK_ONLY:-0}"
COLLECT_ONLY="${COLLECT_ONLY:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_EQUIVALENT_EXISTING="${SKIP_EQUIVALENT_EXISTING:-1}"

mkdir -p "$LOG_ROOT" "$RESULTS_ROOT"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "CCA-MIL ablation sweep"
echo "  python: $PYTHON_BIN"
echo "  run: $RUN_NAME"
echo "  datasets: $DATASETS"
echo "  shots: $SHOTS"
echo "  folds: $FOLDS"
echo "  epochs: $MAX_EPOCHS"
echo "  preset: $PRESET"
echo "  gpus: $GPUS"
echo "  jobs/gpu: $MAX_JOBS_PER_GPU"
echo "  rank metric: $RANK_METRIC"
echo "  skip existing: $SKIP_EXISTING"
echo "  skip equivalent existing: $SKIP_EQUIVALENT_EXISTING"

SWEEP_ARGS=(
  --datasets "$DATASETS"
  --shots "$SHOTS"
  --seeds "$SEEDS"
  --preset "$PRESET"
  --run-name "$RUN_NAME"
  --results-root "$RESULTS_ROOT/AOT_MIL_sweeps"
  --logs-root "$LOG_ROOT"
  --gpus "$GPUS"
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU"
  --folds "$FOLDS"
  --max-epochs "$MAX_EPOCHS"
  --python "$PYTHON_BIN"
  --rank-metric "$RANK_METRIC"
)

if [[ "$CHECK_ONLY" == "1" ]]; then
  SWEEP_ARGS+=(--validate-only)
fi

if [[ "$COLLECT_ONLY" == "1" ]]; then
  SWEEP_ARGS+=(--collect-only)
fi

if [[ "$SKIP_EXISTING" != "1" ]]; then
  SWEEP_ARGS+=(--no-skip-existing)
fi

if [[ "$SKIP_EQUIVALENT_EXISTING" != "1" ]]; then
  SWEEP_ARGS+=(--no-skip-equivalent-existing)
fi

cd "$ROOT_DIR"
"$PYTHON_BIN" tools/run_aot_sweep.py "${SWEEP_ARGS[@]}"

if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "CHECK_ONLY=1, no result CSVs were collected."
  exit 0
fi

"$PYTHON_BIN" tools/collect_ablation_results.py \
  --root "$RESULTS_ROOT" \
  --output "$RESULTS_ROOT/ablation_results_all.csv" \
  --runs "$RUN_NAME" \
  --datasets "$(echo "$DATASETS" | sed 's/all//')" \
  --shots "$SHOTS" \
  --sort-by "$RANK_METRIC" \
  --libra-summary "$LIBRA_SUMMARY" \
  --best-output "$RESULTS_ROOT/CCA_MIL_${RUN_NAME}_best_vs_Libra.csv" \
  --print-top 20

echo "Done."
echo "  all ablations: $RESULTS_ROOT/ablation_results_all.csv"
echo "  best vs Libra: $RESULTS_ROOT/CCA_MIL_${RUN_NAME}_best_vs_Libra.csv"
