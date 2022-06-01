
dataset='SHL' #
trj_branch_method='denoising_imputation_pretrain'
feature_branch_method='denoising_pretrain'

dir_prefix='512night'

pre_epoch=100
finetune_epoch=150

python src/main.py --output_dir experiments --comment "trj ${trj_branch_method}" --name ${dir_prefix}_${dataset}_trj_${trj_branch_method} --task ${trj_branch_method} --records_file trj_${trj_branch_method}.xls --data_class trajectory --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${pre_epoch} --lr 0.001 --optimizer RAdam --batch_size 1000 --pos_encoding learnable --num_workers 16 --d_model 64  --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix
sleep 5s
python src/main.py --output_dir experiments --comment "feat ${feature_branch_method}" --name ${dir_prefix}_${dataset}_feature_${feature_branch_method} --task ${feature_branch_method} --records_file feat_${feature_branch_method}.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${pre_epoch} --lr 0.001 --optimizer RAdam --batch_size 1000 --pos_encoding learnable --d_model 64 --num_workers 16 --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix
sleep 5s
python src/main.py --output_dir experiments --comment "finetune for classification" --name ${dir_prefix}_${dataset}_dual_branch_finetune_trj_${trj_branch_method}_feat_${feature_branch_method} --task dual_branch_classification --records_file Classification_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${finetune_epoch} --lr 0.001 --optimizer RAdam --batch_size 550 --pos_encoding learnable --num_workers 16  --input_type mix --change_output --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_feature_branch experiments/tmp/feature_model_best.pth --load_trajectory_branch experiments/tmp/trajectory_model_best.pth
sleep 5s

#---
trj_branch_method='denoising_pretrain'
feature_branch_method='denoising_pretrain'

python src/main.py --output_dir experiments --comment "trj ${trj_branch_method}" --name ${dir_prefix}_${dataset}_trj_${trj_branch_method} --task ${trj_branch_method} --records_file trj_${trj_branch_method}.xls --data_class trajectory --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${pre_epoch} --lr 0.001 --optimizer RAdam --batch_size 1000 --pos_encoding learnable --num_workers 16 --d_model 64  --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix
sleep 5s
python src/main.py --output_dir experiments --comment "feat ${feature_branch_method}" --name ${dir_prefix}_${dataset}_feature_${feature_branch_method} --task ${feature_branch_method} --records_file feat_${feature_branch_method}.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${pre_epoch} --lr 0.001 --optimizer RAdam --batch_size 1000 --pos_encoding learnable --d_model 64 --num_workers 16 --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix
sleep 5s
python src/main.py --output_dir experiments --comment "finetune for classification" --name ${dir_prefix}_${dataset}_dual_branch_finetune_trj_${trj_branch_method}_feat_${feature_branch_method} --task dual_branch_classification --records_file Classification_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${finetune_epoch} --lr 0.001 --optimizer RAdam --batch_size 550 --pos_encoding learnable --num_workers 16  --input_type mix --change_output --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_feature_branch experiments/tmp/feature_model_best.pth --load_trajectory_branch experiments/tmp/trajectory_model_best.pth
sleep 5s

#---
trj_branch_method='denoising_imputation_pretrain'
feature_branch_method='denoising_imputation_pretrain'

python src/main.py --output_dir experiments --comment "trj ${trj_branch_method}" --name ${dir_prefix}_${dataset}_trj_${trj_branch_method} --task ${trj_branch_method} --records_file trj_${trj_branch_method}.xls --data_class trajectory --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${pre_epoch} --lr 0.001 --optimizer RAdam --batch_size 1000 --pos_encoding learnable --num_workers 16 --d_model 64  --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix
sleep 5s
python src/main.py --output_dir experiments --comment "feat ${feature_branch_method}" --name ${dir_prefix}_${dataset}_feature_${feature_branch_method} --task ${feature_branch_method} --records_file feat_${feature_branch_method}.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${pre_epoch} --lr 0.001 --optimizer RAdam --batch_size 1000 --pos_encoding learnable --d_model 64 --num_workers 16 --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix
sleep 5s
python src/main.py --output_dir experiments --comment "finetune for classification" --name ${dir_prefix}_${dataset}_dual_branch_finetune_trj_${trj_branch_method}_feat_${feature_branch_method} --task dual_branch_classification --records_file Classification_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${finetune_epoch} --lr 0.001 --optimizer RAdam --batch_size 550 --pos_encoding learnable --num_workers 16  --input_type mix --change_output --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_feature_branch experiments/tmp/feature_model_best.pth --load_trajectory_branch experiments/tmp/trajectory_model_best.pth
sleep 5s















#-----
#python src/main.py --output_dir experiments --comment "pretraining through trj_denoising_imputation" --name ${dataset}_trj_pretrained_only_denoi --task denoising_pretrain --records_file Denoising_imputation_records.xls --data_class trajectory --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 10 --lr 0.001 --optimizer RAdam --batch_size 900 --pos_encoding learnable --num_workers 16 --d_model 64  --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix
#sleep 5s
#python src/main.py --output_dir experiments --comment "pretraining through feat_denoising" --name ${dataset}_feature_pretrained_only_denoi --task denoising_pretrain --records_file Denoising_imputation_records.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 50 --lr 0.001 --optimizer RAdam --batch_size 900 --pos_encoding learnable --d_model 64 --num_workers 16 --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix
#sleep 5s
#python src/main.py --output_dir experiments --comment "finetune for classification" --name ${dataset}_dual_branch_finetune_trj_denoi_mf_denoi --task dual_branch_classification --records_file Classification_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 100 --lr 0.001 --optimizer RAdam --batch_size 550 --pos_encoding learnable --num_workers 16  --input_type mix --change_output --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_feature_branch experiments/tmp/feature_model_best.pth --load_trajectory_branch experiments/tmp/trajectory_model_best.pth

#test
#python src/main.py --output_dir test_res --comment "finetune for classification" --name ${dataset}_dual_branch_finetune_test --task dual_branch_classification --records_file Classification_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 200 --lr 0.001 --optimizer RAdam --batch_size 400 --pos_encoding learnable --num_workers 16 --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_model experiments/geolife_sub_dual_branch_finetune_trj_denoi_mf_denoi_2022-05-11_14-03-07_Yj9/checkpoints/model_best.pth  --input_type noise --test_only testset

#tmp
#python src/main.py --output_dir experiments --comment "finetune for classification" --name ${dataset}_dual_branch_finetune_freeze --task dual_branch_classification --records_file Classification_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 100 --lr 0.001 --optimizer RAdam --batch_size 450 --pos_encoding learnable --num_workers 16  --input_type mix --change_output --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_feature_branch experiments/geolife_feature_pretrained_2022-05-10_13-31-55_g5t/checkpoints/model_best.pth --load_trajectory_branch experiments/geolife_trj_pretrained_2022-05-10_13-25-17_ezt/checkpoints/model_best.pth --freeze

#python src/main.py --output_dir experiments --comment "finetune for classification" --name ${dataset}_dual_branch_finetune_trj_denoi_mf_denoi --task dual_branch_classification --records_file Classification_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 100 --lr 0.001 --optimizer RAdam --batch_size 600 --pos_encoding learnable --num_workers 16  --input_type mix --change_output --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_feature_branch experiments/geolife_sub_feature_pretrained_denoi_imp_2022-05-11_13-57-08_6Af/checkpoints/model_best.pth --load_trajectory_branch experiments/geolife_sub_trj_pretrained_denoi_imp_2022-05-11_10-52-05_FYY/checkpoints/model_best.pth
#
#
#python src/main.py --output_dir experiments --comment "pretraining through feat_denoising_imputation" --name ${dataset}_feature_pretrained --task denoising_imputation_pretrain --records_file Denoising_imputation_records.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 50 --lr 0.001 --optimizer RAdam --batch_size 1000 --pos_encoding learnable --d_model 64 --num_workers 16 --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix

