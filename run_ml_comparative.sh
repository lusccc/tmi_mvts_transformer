dataset=('SHL' 'geolife')
# dataset=('SHL')
method=('ml_classification')
train_input_type='50%noise'
test_input_type=('0%noise' '10%noise' '20%noise' '30%noise' '40%noise' '50%noise' '60%noise' '70%noise' '80%noise' '90%noise' '100%noise')

batch_size=2000  # 添加batch_size变量

# 遍历dataset、method的组合进行训练，每训练完一个就遍历test_input_type进行测试
for d in "${dataset[@]}"; do
    for m in "${method[@]}"; do
        # 为训练生成时间戳和目录名
        train_timestamp=$(date +"%Y%m%d_%H%M%S")
        train_dir_name="${d}_${m}"
        train_experiment_name="${train_dir_name}_comparative_exp"
        train_output_dir="train_res/${train_timestamp}_${train_dir_name}"
        
        # 创建训练输出目录
        mkdir -p "${train_output_dir}/checkpoints"
        
        echo "开始训练: ${d} - ${m}"
        # 训练命令
        python main.py --output_dir ${train_output_dir} --experiment_name ${train_experiment_name} \
            --task ${m} --data_class feature \
            --data_name ${d} --val_ratio 0.1 --batch_size ${batch_size} \
            --num_workers 24 --input_type ${train_input_type} \
            --motion_features "0,1,2,3,4,5,6,7,8"  # 会在ml_classification.py中进行特征选择
        
        # 检查训练命令是否执行成功
        if [ $? -ne 0 ]; then
            echo "训练过程出错，退出脚本"
            exit 1
        fi
        
        # 训练完成后，遍历不同噪声级别进行测试
        for k in "${!test_input_type[@]}"; do
            # 为测试生成时间戳和目录名
            test_timestamp=$(date +"%Y%m%d_%H%M%S")
            test_dir_name="${d}_${m}_${test_input_type[$k]}"
            test_experiment_name="${test_dir_name}_test"
            test_output_dir="test_res/${test_timestamp}_${test_dir_name}"
            
            # 创建测试输出目录
            mkdir -p "${test_output_dir}"
            
            echo "开始测试: ${d} - ${m} - ${test_input_type[$k]}"
            # 测试命令
            python main.py --output_dir ${test_output_dir} --experiment_name ${test_experiment_name} \
                 --task ${m} --data_class feature \
                 --data_name ${d} --val_ratio 0.1 --batch_size ${batch_size} \
                 --num_workers 24 --load_model ${train_output_dir}/checkpoints/ml_models \
                 --input_type ${test_input_type[$k]} --test_only testset \
                 --motion_features "0,1,2,3,4,5,6,7,8"
            
            # 检查测试命令是否执行成功
            if [ $? -ne 0 ]; then
                echo "测试过程出错，退出脚本"
                exit 1
            fi
        done
    done
done