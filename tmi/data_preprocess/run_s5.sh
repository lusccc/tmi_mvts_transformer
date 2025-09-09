#!/bin/bash

dataset='SHL'

# 清理之前的噪声数据
rm -rf ../../data/${dataset}_features_sim_noise/train/
rm -rf ../../data/${dataset}_features_sim_noise/test/

echo "Starting GPS noise simulation for ${dataset} dataset..."

# # 对训练集施加噪声
# echo "Processing training set..."
# python s5_realistic_diverse_noise_simulation.py \
#     --trjs_path ../../data/${dataset}_augmented/train_trjs_augmented.npy \
#     --labels_path ../../data/${dataset}_augmented/train_labels_augmented.npy \
#     --n_class 5 \
#     --save_dir ../../data/${dataset}_features_sim_noise/train/

# 对测试集施加噪声
echo "Processing test set..."
python s5_realistic_diverse_noise_simulation.py \
    --trjs_path ../../data/${dataset}_split/test_trjs.npy \
    --labels_path ../../data/${dataset}_split/test_labels.npy \
    --n_class 5 \
    --save_dir ../../data/${dataset}_features_sim_noise/test/

echo "GPS noise simulation completed!"
echo "Results saved to:"
echo "  - Train: ../../data/${dataset}_features_sim_noise/train/"
echo "  - Test:  ../../data/${dataset}_features_sim_noise/test/"
echo ""
echo "Generated noise configurations for both train and test:"
echo "- drift_10_percent, drift_20_percent, drift_30_percent"
echo "- missing_10_percent, missing_20_percent, missing_30_percent" 
echo "- mixed_10_percent, mixed_20_percent, mixed_30_percent"
echo "- clean_original (for comparison)"