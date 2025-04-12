datasets=('SHL' 'geolife')
branch=trajectory
d_model=64
num_heads=8
num_layers=4
dim_feedforward=256
pos_encoding=fixed
normalization_layer=BatchNorm

batch_size=600
n_patience=30

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: ${dataset}"
    for pretrain_epoch in 2 5 10 20 50 100 150 200; do
        timestamp=$(date +"%Y%m%d_%H%M%S")
        pretrain_exp_prefix="${dataset}_${branch}_pretrain"
        pretrain_experiment_name="${pretrain_exp_prefix}_d${d_model}_h${num_heads}_l${num_layers}_f${dim_feedforward}_${pos_encoding}_${normalization_layer}_epoch${pretrain_epoch}"
        pretrain_output_dir="train_res_${pretrain_exp_prefix}/${timestamp}_${pretrain_experiment_name}"
        pretrain_records_file="${pretrain_output_dir}/records.xlsx"

        echo '---------pretrain model-----------------'
        #  !!! 注意val_ratio=0 不用验证集，那么也没有早停
        python main.py \
            --output_dir ${pretrain_output_dir} \
            --experiment_name ${pretrain_experiment_name} \
            --task denoising_imputation_pretrain \
            --records_file ${pretrain_records_file} \
            --data_class ${branch} \
            --data_name ${dataset} \
            --val_ratio 0 \
            --epochs ${pretrain_epoch} \
            --batch_size ${batch_size} \
            --num_workers 56 \
            --input_type "50%noise" \
            --pos_encoding ${pos_encoding} \
            --d_model ${d_model} \
            --num_heads ${num_heads} \
            --num_layers ${num_layers} \
            --dim_feedforward ${dim_feedforward} \
            --normalization_layer ${normalization_layer}

        if [ $? -ne 0 ]; then
            echo "预训练阶段执行失败，退出脚本"
            exit 1
        fi

        echo '---------fintune model-----------------'
        timestamp=$(date +"%Y%m%d_%H%M%S")
        fintune_exp_prefix="${dataset}_${branch}_fintune"
        fintune_experiment_name="${fintune_exp_prefix}_d${d_model}_h${num_heads}_l${num_layers}_f${dim_feedforward}_${pos_encoding}_${normalization_layer}_epoch${pretrain_epoch}"
        fintune_output_dir="train_res_${fintune_exp_prefix}/${timestamp}_${fintune_experiment_name}"
        fintune_records_file="${fintune_output_dir}/records.xlsx"
        python main.py \
            --output_dir ${fintune_output_dir} \
            --experiment_name ${fintune_experiment_name} \
            --task feature_branch_classification \
            --records_file ${fintune_records_file} \
            --data_class ${branch} \
            --data_name ${dataset} \
            --val_ratio 0.1 \
            --epochs 200 \
            --batch_size ${batch_size} \
            --num_workers 24 \
            --patience ${n_patience} \
            --input_type "50%noise" \
            --feature_branch_hyperparams "${pretrain_output_dir}/denoising_imputation_pretrain_model_hyperparams.json" \
            --load_model "${pretrain_output_dir}/checkpoints/model_last.pth" \
            --change_output

        if [ $? -ne 0 ]; then
            echo "微调阶段执行失败，退出脚本"
            exit 1
        fi
    done
done