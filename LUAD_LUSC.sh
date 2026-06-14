log_dir='logs/'
task='LUAD_LUSC'
shots=16
folds=10
model='CCA_MIL'
feature='conch'
device=0
dataset_csv='/home/yuhaowang/project/WSIFew/cca_mil/dataset_csv/LUAD_LUSC.csv'
feature_dir='/data/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature/pt_files'

mkdir -p $log_dir
export CUDA_VISIBLE_DEVICES=$device
exp=$model"/"$feature
echo "Task: "$task", Shots: "$shots", "$exp", GPU No.:"$device
nohup python main.py \
    --seed 1 \
    --drop_out \
    --early_stopping \
    --lr 1e-4 \
    --k $folds \
    --label_frac 1 \
    --bag_loss ce \
    --task "task_tcga_lung_subtyping" \
    --csv_path $dataset_csv \
    --results_dir 'results/'$model'/'$feature'/' \
    --exp_code $task"_"$shots"shots_"$folds"folds" \
    --model_type $model \
    --mode transformer \
    --log_data \
    --data_root_dir '/data/yuhaowang/WSIFew' \
    --data_folder_s $feature_dir \
    --data_folder_l $feature_dir \
    --split_dir $task"_"$shots"shots_"$folds"folds" \
    --concept_bank_path 'text_prompt/concept_bank/tcga_nsclc.json' \
    --cluster_k 8 \
    --selection_top_r 3 \
    --concept_alpha 0.5 \
    --lambda_con 0.1 \
    --lambda_div 0.01 \
    --prototype_number 16 > $log_dir$task"_"$model"_"$shots"shots_"$folds"folds_"$feature".log" 2>&1 &
