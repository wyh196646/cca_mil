log_dir='logs/'
task='UBC-OCEAN'
shots=16
folds=10
model='CCA_MIL'
feature='conch'
device=0

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
    --task "task_UBC-OCEAN_subtyping" \
    --results_dir 'results/'$model'/'$feature'/' \
    --exp_code $task"_"$shots"shots_"$folds"folds" \
    --model_type $model \
    --mode transformer \
    --log_data \
    --data_root_dir 'path/to/your/data' \
    --data_folder_s 'path/to/your/low-resolution/feature' \
    --data_folder_l 'path/to/your/high-resolution/feature' \
    --split_dir $task"_"$shots"shots_"$folds"folds" \
    --concept_bank_path 'text_prompt/concept_bank/ubc_ocean.json' \
    --cluster_k 8 \
    --selection_top_r 3 \
    --concept_alpha 0.5 \
    --lambda_con 0.1 \
    --lambda_div 0.01 \
    --prototype_number 16 > $log_dir$task"_"$model"_"$shots"shots_"$folds"folds_"$feature".txt" 2>&1 &
