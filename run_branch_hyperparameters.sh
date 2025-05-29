dataset='geolife'
branch=('trajectory')

# 遍历所有分支
for current_branch in "${branch[@]}"; do
    task="${current_branch}_branch_classification_from_scratch"
    exp_prefix="${current_branch}_branch_hyperparameters"

    batch_size=80
    epochs=250
    n_patience=40

    # 遍历所有超参数组合
    # for d_model in 64 128 256 512; do
    for d_model in 512; do
        # 设置dim_feedforward为d_model的4倍
        dim_feedforward=$((d_model * 4))
        
        for num_heads in 4 8 16 32; do
            # for num_layers in 1 2 3 4 6; do
            for num_layers in 1 2 3 4; do
                # for pos_encoding in learnable fixed; do
                for pos_encoding in fixed; do  # TODO trajectory 使用fixed, feature 使用learnable
                    # for normalization_layer in BatchNorm LayerNorm; do
                    for normalization_layer in LayerNorm; do
                        timestamp=$(date +"%Y%m%d_%H%M%S")
                        experiment_name="${exp_prefix}_d${d_model}_h${num_heads}_l${num_layers}_f${dim_feedforward}_${pos_encoding}_${normalization_layer}"
                        output_dir="train_res_hpsearch_${dataset}_${current_branch}/${timestamp}_${experiment_name}"
                        records_file="${output_dir}/records.xlsx"

                        # 检查是否已经运行过相同参数组合的噪声扫描测试
                        test_exp_prefix="${current_branch}_branch_noise_sweep"
                        noise_result_pattern="test_res_hpsearch_${dataset}_${current_branch}/${test_exp_prefix}/*_${test_exp_prefix}_d${d_model}_h${num_heads}_l${num_layers}_f${dim_feedforward}_${pos_encoding}_${normalization_layer}/noise_level_sweep_results.xlsx"
                        if ls $noise_result_pattern 1> /dev/null 2>&1; then
                            echo "跳过已运行过的参数组合: d_model=${d_model}, heads=${num_heads}, layers=${num_layers}, pos_encoding=${pos_encoding}, normalization_layer=${normalization_layer}"
                            continue
                        fi

                        echo '--------------------------------'
                        echo "运行实验: d_model=${d_model}, dim_feedforward=${dim_feedforward} (4×d_model), heads=${num_heads}, layers=${num_layers}"
                        python main.py \
                            --output_dir ${output_dir} \
                            --experiment_name ${experiment_name} \
                            --task ${task} \
                            --records_file ${records_file} \
                            --data_class ${current_branch} \
                            --data_name ${dataset} \
                            --val_ratio 0.1 \
                            --epochs ${epochs} \
                            --batch_size ${batch_size} \
                            --num_workers 24 \
                            --patience ${n_patience} \
                            --input_type "50%noise" \
                            --pos_encoding ${pos_encoding} \
                            --normalization_layer ${normalization_layer} \
                            --d_model ${d_model} \
                            --num_heads ${num_heads} \
                            --num_layers ${num_layers} \
                            --dim_feedforward ${dim_feedforward}
                            
                        # 添加测试噪声级别的实验运行
                        test_timestamp=$(date +"%Y%m%d_%H%M%S")
                        test_exp_prefix="${current_branch}_branch_noise_sweep"
                        test_experiment_name="${test_exp_prefix}_d${d_model}_h${num_heads}_l${num_layers}_f${dim_feedforward}_${pos_encoding}_${normalization_layer}"
                        test_output_dir="test_res_hpsearch_${dataset}_${current_branch}/${test_timestamp}_${test_experiment_name}"
                        test_records_file="${test_output_dir}/records.xlsx"
                        
                        # 设置分支参数
                        hyperparams_file="${output_dir}/${task}_model_hyperparams.json"
                        if [ "$current_branch" == "feature" ]; then
                            branch_param="--feature_branch_hyperparams ${hyperparams_file}"
                        else
                            branch_param="--trajectory_branch_hyperparams ${hyperparams_file}"
                        fi
                        
                        echo '--------- 开始噪声级别扫描测试 ---------'
                        echo "使用模型: ${output_dir}/checkpoints/model_best.pth"
                        echo "使用超参数文件: ${hyperparams_file}"
                        echo "输出目录: ${test_output_dir}"
                        python main.py \
                            --output_dir ${test_output_dir} \
                            --experiment_name ${test_experiment_name} \
                            --task ${task} \
                            --records_file ${test_records_file} \
                            --data_class ${current_branch} \
                            --data_name ${dataset} \
                            --val_ratio 0.1 \
                            --batch_size ${batch_size} \
                            --num_workers 24 \
                            ${branch_param} \
                            --load_model "${output_dir}/checkpoints/model_best.pth" \
                            --test_only testset \
                            --noise_level_sweep
                    done
                done
            done
        done
    done
done

