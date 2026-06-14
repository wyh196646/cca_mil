#!/usr/bin/env bash
set -euo pipefail

log_dir='logs/'
task='LUAD_LUSC'
shots="${SHOTS:-16}"
folds="${FOLDS:-10}"
model='FOCUS'
feature='conch'
device="${DEVICE:-0}"

dataset_csv="${DATASET_CSV:-/home/yuhaowang/project/WSIFew/cca_mil/dataset_csv/LUAD_LUSC.csv}"
feature_dir="${FEATURE_DIR:-/data/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature/pt_files}"
text_prompt_path="${TEXT_PROMPT_PATH:-text_prompt/TCGA_Lung_two_scale_text_prompt.csv}"
conch_ckpt_path="${CONCH_CKPT_PATH:-/home/yuhaowang/project/WSIFew/cca_mil/ckg/pytorch_model.bin}"

mkdir -p "$log_dir"
export CUDA_VISIBLE_DEVICES="$device"

extra_args=()
if [[ "${K_START:-}" != "" ]]; then
  extra_args+=(--k_start "$K_START")
fi
if [[ "${K_END:-}" != "" ]]; then
  extra_args+=(--k_end "$K_END")
fi

exp=$model"/"$feature
echo "Task: $task, Shots: $shots, $exp, GPU No.:$device"

python main.py \
    --seed 1 \
    --drop_out \
    --early_stopping \
    --lr 1e-4 \
    --k "$folds" \
    --label_frac 1 \
    --bag_loss ce \
    --task "task_tcga_lung_subtyping" \
    --csv_path "$dataset_csv" \
    --results_dir "results/$model/$feature/" \
    --exp_code "${task}_${shots}shots_${folds}folds" \
    --model_type "$model" \
    --mode transformer \
    --log_data \
    --data_root_dir '/data/yuhaowang/WSIFew' \
    --data_folder_s "$feature_dir" \
    --data_folder_l "$feature_dir" \
    --split_dir "${task}_${shots}shots_${folds}folds" \
    --text_prompt_path "$text_prompt_path" \
    --conch_ckpt_path "$conch_ckpt_path" \
    --max_context_length 8192 \
    --window_size 8 \
    --sim_threshold 0.8 \
    --prototype_number 16 \
    "${extra_args[@]}" \
    2>&1 | tee "${log_dir}${task}_${model}_${shots}shots_${folds}folds_${feature}.log"
