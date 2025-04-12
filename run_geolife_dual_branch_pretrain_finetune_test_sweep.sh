#!/bin/bash
set -e  # 遇到错误时退出

# 数据集配置
datasets=('geolife')

# 特征分支配置
feature_d_model=128
feature_num_heads=16
feature_num_layers=1
feature_dim_feedforward=512
feature_pos_encoding=learnable
feature_normalization_layer=LayerNorm
feature_pretrain_epoch=250  # 特征分支预训练轮数

# 轨迹分支配置
trajectory_d_model=64
trajectory_num_heads=8
trajectory_num_layers=4
trajectory_dim_feedforward=256
trajectory_pos_encoding=fixed
trajectory_normalization_layer=BatchNorm
trajectory_pretrain_epoch=100  # 轨迹分支预训练轮数

# 融合配置
finetune_epoch=300  # 微调轮数
batch_size=1200
n_patience=40

# 遍历处理每个数据集
for dataset in "${datasets[@]}"; do
    # 1. 特征分支预训练阶段
    feature_pretrain_exp_prefix="${dataset}_feature_pretrain"
    feature_pretrain_experiment_name="${feature_pretrain_exp_prefix}_d${feature_d_model}_h${feature_num_heads}_l${feature_num_layers}_f${feature_dim_feedforward}_${feature_pos_encoding}_${feature_normalization_layer}_epoch${feature_pretrain_epoch}"
    feature_pretrain_output_dir="train_res/${feature_pretrain_exp_prefix}/${feature_pretrain_experiment_name}"
    feature_pretrain_records_file="${feature_pretrain_output_dir}/records.xlsx"

    python -u main.py \
        --output_dir ${feature_pretrain_output_dir} \
        --experiment_name ${feature_pretrain_experiment_name} \
        --task denoising_imputation_pretrain \
        --records_file ${feature_pretrain_records_file} \
        --data_class feature \
        --data_name ${dataset} \
        --val_ratio 0 \
        --epochs ${feature_pretrain_epoch} \
        --batch_size ${batch_size} \
        --num_workers 24 \
        --input_type "50%noise" \
        --pos_encoding ${feature_pos_encoding} \
        --d_model ${feature_d_model} \
        --num_heads ${feature_num_heads} \
        --num_layers ${feature_num_layers} \
        --dim_feedforward ${feature_dim_feedforward} \
        --normalization_layer ${feature_normalization_layer}

    # 2. 轨迹分支预训练阶段
    trajectory_pretrain_exp_prefix="${dataset}_trajectory_pretrain"
    trajectory_pretrain_experiment_name="${trajectory_pretrain_exp_prefix}_d${trajectory_d_model}_h${trajectory_num_heads}_l${trajectory_num_layers}_f${trajectory_dim_feedforward}_${trajectory_pos_encoding}_${trajectory_normalization_layer}_epoch${trajectory_pretrain_epoch}"
    trajectory_pretrain_output_dir="train_res/${trajectory_pretrain_exp_prefix}/${trajectory_pretrain_experiment_name}"
    trajectory_pretrain_records_file="${trajectory_pretrain_output_dir}/records.xlsx"

    python -u main.py \
        --output_dir ${trajectory_pretrain_output_dir} \
        --experiment_name ${trajectory_pretrain_experiment_name} \
        --task denoising_imputation_pretrain \
        --records_file ${trajectory_pretrain_records_file} \
        --data_class trajectory \
        --data_name ${dataset} \
        --val_ratio 0 \
        --epochs ${trajectory_pretrain_epoch} \
        --batch_size ${batch_size} \
        --num_workers 24 \
        --input_type "50%noise" \
        --pos_encoding ${trajectory_pos_encoding} \
        --d_model ${trajectory_d_model} \
        --num_heads ${trajectory_num_heads} \
        --num_layers ${trajectory_num_layers} \
        --dim_feedforward ${trajectory_dim_feedforward} \
        --normalization_layer ${trajectory_normalization_layer}

    # 3. 双分支融合微调阶段
    finetune_batch_size=600
    finetune_exp_prefix="${dataset}_dual_branch_finetune"
    finetune_experiment_name="${finetune_exp_prefix}_feature_d${feature_d_model}_trajectory_d${trajectory_d_model}"
    finetune_output_dir="train_res/${finetune_exp_prefix}/${finetune_experiment_name}"
    finetune_records_file="${finetune_output_dir}/records.xlsx"
    
    python -u main.py \
        --output_dir ${finetune_output_dir} \
        --experiment_name ${finetune_experiment_name} \
        --task dual_branch_classification \
        --records_file ${finetune_records_file} \
        --data_class trajectory_with_feature \
        --data_name ${dataset} \
        --val_ratio 0.1 \
        --epochs ${finetune_epoch} \
        --batch_size ${finetune_batch_size} \
        --num_workers 24 \
        --patience ${n_patience} \
        --input_type "50%noise" \
        --feature_branch_hyperparams "${feature_pretrain_output_dir}/denoising_imputation_pretrain_model_hyperparams.json" \
        --trajectory_branch_hyperparams "${trajectory_pretrain_output_dir}/denoising_imputation_pretrain_model_hyperparams.json" \
        --load_feature_branch "${feature_pretrain_output_dir}/checkpoints/model_last.pth" \
        --load_trajectory_branch "${trajectory_pretrain_output_dir}/checkpoints/model_last.pth" \
        --change_output

    # 4. 测试阶段
    test_exp_prefix="${dataset}_dual_branch_noise_sweep"
    test_experiment_name="${test_exp_prefix}_feature_d${feature_d_model}_trajectory_d${trajectory_d_model}"
    test_output_dir="test_res/${test_exp_prefix}/${test_experiment_name}"
    test_records_file="${test_output_dir}/records.xlsx"
    
    python -u main.py \
        --output_dir ${test_output_dir} \
        --experiment_name ${test_experiment_name} \
        --task dual_branch_classification \
        --records_file ${test_records_file} \
        --data_class trajectory_with_feature \
        --data_name ${dataset} \
        --val_ratio 0.1 \
        --batch_size ${finetune_batch_size} \
        --num_workers 24 \
        --feature_branch_hyperparams "${feature_pretrain_output_dir}/denoising_imputation_pretrain_model_hyperparams.json" \
        --trajectory_branch_hyperparams "${trajectory_pretrain_output_dir}/denoising_imputation_pretrain_model_hyperparams.json" \
        --load_model "${finetune_output_dir}/checkpoints/model_best.pth" \
        --test_only testset \
        --noise_level_sweep
done
