dataset='geolife'

rm -rf ../../data/${dataset}_features/train/
rm -rf ../../data/${dataset}_features/test/

bw=1
kernel='epa'
mask_mode='ep'  
trj_mask_mode='kde'
mask_ratio=0.3


python s4_trajectory_feature_calculation_with_CPD.py --trjs_path ../../data/${dataset}_augmented/train_trjs_augmented.npy --labels_path ../../data/${dataset}_augmented/train_labels_augmented.npy --n_class 5 --save_dir ../../data/${dataset}_features/train/ --kde_bw $bw --kde_kernel $kernel  --mask_mode $mask_mode --trj_mask_mode $trj_mask_mode --mask_ratio $mask_ratio

python s4_trajectory_feature_calculation_with_CPD.py --trjs_path ../../data/${dataset}_split/test_trjs.npy --labels_path ../../data/${dataset}_split/test_labels.npy --n_class 5 --save_dir ../../data/${dataset}_features/test/ --kde_bw $bw --kde_kernel $kernel --mask_mode $mask_mode --trj_mask_mode $trj_mask_mode --mask_ratio $mask_ratio