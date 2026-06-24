#!/usr/bin/env bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_root=${OUTPUT_ROOT:-/data2/yuhaowang/cca-mil-result}
log_dir="$output_root/logs/"
task='TCGA_RCC'
shots=${SHOTS:-16}
folds=${FOLDS:-5}
max_epochs=${MAX_EPOCHS:-80}
model='CCA_MIL'
feature='conch'
device=${GPU:-0}
python_bin=${PYTHON_BIN:-python}
dataset_csv="$ROOT_DIR/dataset_csv/RCC.csv"
feature_dir='/data2/yuhaowang/WSIFew/processd_wsi/TCGA-RCC/feature/pt_files'
concept_bank='text_prompt/concept_bank/tcga_rcc.json'

mkdir -p "$log_dir"
export CUDA_VISIBLE_DEVICES=$device
cd "$ROOT_DIR"
exp=$model"/"$feature
echo "Task: "$task", Shots: "$shots", "$exp", GPU No.:"$device
nohup "$python_bin" main.py \
    --seed 1 \
    --drop_out \
    --early_stopping \
    --early_stopping_patience 15 \
    --early_stopping_stop_epoch 0 \
    --max_epochs $max_epochs \
    --lr 1e-4 \
    --k $folds \
    --label_frac 1 \
    --bag_loss ce \
    --task "task_tcga_rcc_subtyping" \
    --csv_path "$dataset_csv" \
    --results_dir "$output_root/results/$model/$feature/" \
    --exp_code $task"_"$shots"shots_"$folds"folds" \
    --model_type $model \
    --mode transformer \
    --log_data \
    --data_root_dir '/data2/yuhaowang/WSIFew' \
    --data_folder_s "$feature_dir" \
    --data_folder_l "$feature_dir" \
    --split_dir $task"_"$shots"shots_"$folds"folds" \
    --concept_bank_path "$concept_bank" \
    --num_visual_prototypes 10 \
    --proto_tau 0.1 \
    --ot_epsilon 0.05 \
    --sinkhorn_iter 20 \
    --uot_rho_a 0.5 \
    --uot_rho_b 0.5 \
    --concept_pooling attention \
    --lambda_contrast 0.1 \
    --lambda_div 0.01 > $log_dir$task"_"$model"_"$shots"shots_"$folds"folds_"$feature".log" 2>&1 &
