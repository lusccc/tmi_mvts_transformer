dataset='geolife'

python s4_trajectory_feature_calculation_with_CPD.py --trjs_path ../../data/${dataset}_augmented/train_trjs_augmented.npy --labels_path ../../data/${dataset}_augmented/train_labels_augmented.npy --n_class 5 --save_dir ../../data/${dataset}_features/train/

python s4_trajectory_feature_calculation_with_CPD.py --trjs_path ../../data/${dataset}_split/test_trjs.npy --labels_path ../../data/${dataset}_split/test_labels.npy --n_class 5 --save_dir ../../data/${dataset}_features/test/