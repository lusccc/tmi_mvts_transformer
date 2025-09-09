dataset=('SHL' 'geolife')
# dataset=('SHL')
# method=('cnn_classification' 'lstm_classification' 'dmn_classification' 'msaf_classification')  # msaf_classification 效果不好弃用
method=('cnn_classification' )
train_input_type='50%noise'

# cnn lstm模型
n_patience=30
epochs=150
batch_size=600

# DMN模型
# n_patience=50
# epochs=300 
# batch_size=1000

# 遍历dataset、method的组合进行训练，每训练完一个就进行一次noise level sweep测试
for d in "${dataset[@]}"; do
    for m in "${method[@]}"; do
        # 为训练生成时间戳和目录名
        train_timestamp=$(date +"%Y%m%d_%H%M%S")
        train_dir_name="${d}_${m}"
        train_experiment_name="${train_dir_name}_comparative_exp"
        train_output_dir="train_res/dl_comparative/${train_timestamp}_${train_dir_name}"
        train_records_file="${train_output_dir}/records.xlsx"
        
        # 创建训练输出目录
        mkdir -p "${train_output_dir}/checkpoints"
        
        echo "开始训练: ${d} - ${m}"
        # 所有方法都使用feature数据类
        data_class="feature"
        
        # 训练命令
        python main.py --output_dir ${train_output_dir} --experiment_name ${train_experiment_name} \
            --task ${m} --records_file ${train_records_file} --data_class ${data_class} \
            --data_name ${d} --val_ratio 0.1 --epochs ${epochs} --batch_size ${batch_size} \
            --num_workers 24 --input_type ${train_input_type} --patience ${n_patience}
        
        # 检查训练命令是否执行成功
        if [ $? -ne 0 ]; then
            echo "训练过程出错，退出脚本"
            exit 1
        fi
        
        # 训练完成后，进行一次noise level sweep测试
        test_timestamp=$(date +"%Y%m%d_%H%M%S")
        test_dir_name="${d}_${m}_noise_sweep"
        test_experiment_name="${test_dir_name}_test"
        test_output_dir="test_res/dl_comparative/${test_timestamp}_${test_dir_name}"
        test_records_file="${test_output_dir}/records.xlsx"
        
        # 创建测试输出目录
        mkdir -p "${test_output_dir}"
        
        echo "开始noise level扫描测试: ${d} - ${m}"
        # 使用新的noise_level_sweep参数进行测试
        python main.py --output_dir ${test_output_dir} --experiment_name ${test_experiment_name} \
            --task ${m} --records_file ${test_records_file} --data_class ${data_class} \
            --data_name ${d} --val_ratio 0.1 --batch_size ${batch_size} \
            --num_workers 24 --load_model ${train_output_dir}/checkpoints/model_best.pth \
            --input_type '50%noise' --test_only testset --noise_level_sweep
        
        # 检查测试命令是否执行成功
        if [ $? -ne 0 ]; then
            echo "测试过程出错，退出脚本"
            exit 1
        fi
    done
done