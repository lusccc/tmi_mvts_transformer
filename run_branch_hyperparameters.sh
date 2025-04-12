dataset='SHL'
branch=('trajectory')

# 遍历所有分支
for current_branch in "${branch[@]}"; do
    task="${current_branch}_branch_classification_from_scratch"
    exp_prefix="${current_branch}_branch_hyperparameters"

    batch_size=600
    epochs=200
    n_patience=20

    # 遍历所有超参数组合
    for d_model in 64 128 256; do
        for num_heads in 2 4 8 16; do
            for num_layers in 1 2 3 4; do
                for dim_feedforward in 64 128 256 512; do
                    for pos_encoding in learnable fixed; do
                        timestamp=$(date +"%Y%m%d_%H%M%S")
                        experiment_name="${exp_prefix}_d${d_model}_h${num_heads}_l${num_layers}_f${dim_feedforward}_${pos_encoding}"
                        output_dir="train_res/${timestamp}_${experiment_name}"
                        records_file="${output_dir}/records.xlsx"

                        echo '--------------------------------'
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
                            --d_model ${d_model} \
                            --num_heads ${num_heads} \
                            --num_layers ${num_layers} \
                            --dim_feedforward ${dim_feedforward}
                    done
                done
            done
        done
    done
done

