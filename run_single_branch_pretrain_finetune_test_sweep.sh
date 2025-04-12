#!/bin/bash

# 数据集和模型配置
datasets=('geolife' )
branch=feature
d_model=128
num_heads=16
num_layers=1
dim_feedforward=512
pos_encoding=learnable
normalization_layer=LayerNorm

# 训练参数
batch_size=600
n_patience=30
pretrain_epoch=150  # 预训练轮数
finetune_epoch=200  # 微调轮数

# 遍历处理每个数据集
for dataset in "${datasets[@]}"; do
    echo "处理数据集: ${dataset}"
    
    # 1. 预训练阶段
    timestamp=$(date +"%Y%m%d_%H%M%S")
    pretrain_exp_prefix="${dataset}_${branch}_pretrain"
    pretrain_experiment_name="${pretrain_exp_prefix}_d${d_model}_h${num_heads}_l${num_layers}_f${dim_feedforward}_${pos_encoding}_${normalization_layer}_epoch${pretrain_epoch}"
    pretrain_output_dir="train_res/${pretrain_exp_prefix}/${timestamp}_${pretrain_experiment_name}"
    pretrain_records_file="${pretrain_output_dir}/records.xlsx"

    echo "===============预训练阶段==============="
    # 注意val_ratio=0不使用验证集，也没有早停
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
        --num_workers 24 \
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

    # 2. 微调阶段
    timestamp=$(date +"%Y%m%d_%H%M%S")
    finetune_exp_prefix="${dataset}_${branch}_finetune"
    finetune_experiment_name="${finetune_exp_prefix}_d${d_model}_h${num_heads}_l${num_layers}_f${dim_feedforward}_${pos_encoding}_${normalization_layer}_epoch${pretrain_epoch}"
    finetune_output_dir="train_res/${finetune_exp_prefix}/${timestamp}_${finetune_experiment_name}"
    finetune_records_file="${finetune_output_dir}/records.xlsx"
    
    echo "===============微调阶段==============="
    python main.py \
        --output_dir ${finetune_output_dir} \
        --experiment_name ${finetune_experiment_name} \
        --task feature_branch_classification \
        --records_file ${finetune_records_file} \
        --data_class ${branch} \
        --data_name ${dataset} \
        --val_ratio 0.1 \
        --epochs ${finetune_epoch} \
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
    
    # 3. 测试阶段 - 使用噪声级别扫描
    timestamp=$(date +"%Y%m%d_%H%M%S")
    test_exp_prefix="${dataset}_${branch}_noise_sweep"
    test_experiment_name="${test_exp_prefix}_d${d_model}_h${num_heads}_l${num_layers}_f${dim_feedforward}_${pos_encoding}_${normalization_layer}"
    test_output_dir="test_res/${test_exp_prefix}/${timestamp}_${test_experiment_name}"
    test_records_file="${test_output_dir}/records.xlsx"
    
    echo "===============噪声扫描测试阶段==============="
    python main.py \
        --output_dir ${test_output_dir} \
        --experiment_name ${test_experiment_name} \
        --task feature_branch_classification \
        --records_file ${test_records_file} \
        --data_class ${branch} \
        --data_name ${dataset} \
        --val_ratio 0.1 \
        --batch_size ${batch_size} \
        --num_workers 24 \
        --feature_branch_hyperparams "${pretrain_output_dir}/denoising_imputation_pretrain_model_hyperparams.json" \
        --load_model "${finetune_output_dir}/checkpoints/model_best.pth" \
        --test_only testset \
        --noise_level_sweep

    if [ $? -ne 0 ]; then
        echo "测试阶段执行失败，退出脚本"
        exit 1
    fi
    
    echo "数据集 ${dataset} 的预训练、微调和测试已完成。"
done

echo "所有数据集处理完毕。" 