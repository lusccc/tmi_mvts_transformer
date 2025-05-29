#!/bin/bash
set -e  # 遇到错误时退出

# 数据集和分支设置
datasets=('geolife' 'SHL')
branches=('feature' 'trajectory')

# 遍历处理每个数据集
for dataset in "${datasets[@]}"; do
    for branch in "${branches[@]}"; do
        # 根据数据集和分支设置不同的超参数
        if [ "$dataset" == "geolife" ] && [ "$branch" == "feature" ]; then
            # Geolife特征分支配置
            d_model=128
            num_heads=16
            num_layers=2
            dim_feedforward=512
            pos_encoding=learnable
            normalization_layer=LayerNorm
            batch_size=600
            n_patience=30
            pretrain_epoch=150  # 预训练轮数
            finetune_epoch=200  # 微调轮数
        elif [ "$dataset" == "geolife" ] && [ "$branch" == "trajectory" ]; then
            # Geolife轨迹分支配置
            d_model=64
            num_heads=8
            num_layers=4
            dim_feedforward=256
            pos_encoding=fixed
            normalization_layer=BatchNorm
            batch_size=600
            n_patience=30
            pretrain_epoch=100  # 预训练轮数
            finetune_epoch=200  # 微调轮数
        elif [ "$dataset" == "SHL" ] && [ "$branch" == "feature" ]; then
            # SHL特征分支配置
            d_model=128
            num_heads=16
            num_layers=1
            dim_feedforward=512
            pos_encoding=learnable
            normalization_layer=LayerNorm
            batch_size=1200
            n_patience=40
            pretrain_epoch=220  # 预训练轮数
            finetune_epoch=300  # 微调轮数
        elif [ "$dataset" == "SHL" ] && [ "$branch" == "trajectory" ]; then
            # SHL轨迹分支配置
            d_model=64
            num_heads=8
            num_layers=4
            dim_feedforward=256
            pos_encoding=fixed
            normalization_layer=BatchNorm
            batch_size=1200
            n_patience=40
            pretrain_epoch=50  # 预训练轮数
            finetune_epoch=300  # 微调轮数
        fi

        echo "开始处理数据集: ${dataset}, 分支: ${branch}"
        echo "配置: d_model=${d_model}, num_heads=${num_heads}, num_layers=${num_layers}, dim_feedforward=${dim_feedforward}"
        echo "位置编码=${pos_encoding}, 归一化层=${normalization_layer}, 批大小=${batch_size}, 预训练轮数=${pretrain_epoch}"
        
        # 1. 预训练阶段
        timestamp=$(date +"%Y%m%d_%H%M%S")
        pretrain_exp_prefix="${dataset}_${branch}_pretrain"
        pretrain_experiment_name="${pretrain_exp_prefix}_d${d_model}_h${num_heads}_l${num_layers}_f${dim_feedforward}_${pos_encoding}_${normalization_layer}_epoch${pretrain_epoch}"
        pretrain_output_dir="train_res/${pretrain_exp_prefix}/${timestamp}_${pretrain_experiment_name}"
        pretrain_records_file="${pretrain_output_dir}/records.xlsx"

        echo "===============预训练阶段==============="
        # 注意val_ratio=0不使用验证集，也没有早停
        python -u main.py \
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
        task_name=""
        if [ "$branch" == "feature" ]; then
            task_name="feature_branch_classification"
        elif [ "$branch" == "trajectory" ]; then
            task_name="trajectory_branch_classification"
        fi
        
        timestamp=$(date +"%Y%m%d_%H%M%S")
        finetune_exp_prefix="${dataset}_${branch}_finetune"
        finetune_experiment_name="${finetune_exp_prefix}_d${d_model}_h${num_heads}_l${num_layers}_f${dim_feedforward}_${pos_encoding}_${normalization_layer}_epoch${pretrain_epoch}"
        finetune_output_dir="train_res/${finetune_exp_prefix}/${timestamp}_${finetune_experiment_name}"
        finetune_records_file="${finetune_output_dir}/records.xlsx"
        
        echo "===============微调阶段==============="
        python -u main.py \
            --output_dir ${finetune_output_dir} \
            --experiment_name ${finetune_experiment_name} \
            --task ${task_name} \
            --records_file ${finetune_records_file} \
            --data_class ${branch} \
            --data_name ${dataset} \
            --val_ratio 0.1 \
            --epochs ${finetune_epoch} \
            --batch_size ${batch_size} \
            --num_workers 24 \
            --patience ${n_patience} \
            --input_type "50%noise" \
            --${branch}_branch_hyperparams "${pretrain_output_dir}/denoising_imputation_pretrain_model_hyperparams.json" \
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
        python -u main.py \
            --output_dir ${test_output_dir} \
            --experiment_name ${test_experiment_name} \
            --task ${task_name} \
            --records_file ${test_records_file} \
            --data_class ${branch} \
            --data_name ${dataset} \
            --val_ratio 0.1 \
            --batch_size ${batch_size} \
            --num_workers 24 \
            --${branch}_branch_hyperparams "${pretrain_output_dir}/denoising_imputation_pretrain_model_hyperparams.json" \
            --load_model "${finetune_output_dir}/checkpoints/model_best.pth" \
            --test_only testset \
            --noise_level_sweep

        if [ $? -ne 0 ]; then
            echo "测试阶段执行失败，退出脚本"
            exit 1
        fi
        
        echo "数据集 ${dataset} 分支 ${branch} 的预训练、微调和测试已完成。"
        echo "========================================================"
    done
done

echo "所有数据集和分支处理完毕！" 