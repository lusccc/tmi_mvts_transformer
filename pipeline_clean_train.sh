dir_prefix='603night'

python src/main.py --output_dir experiments --comment "pretraining through trj_imputation" --name ${dir_prefix}_SHL_trj_pretrained_clean --task denoising_imputation --records_file Denoising_imputation_records.xls --data_class trajectory --data_name SHL --val_ratio 0.1 --test_ratio 0.2 --epochs 50 --lr 0.001 --optimizer RAdam --batch_size 720 --pos_encoding learnable --num_workers 16 --d_model 64  --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type clean

python src/main.py --output_dir experiments --comment "pretraining through feat_denoising_imputation" --name ${dir_prefix}_SHL_feature_pretrained_clean --task denoising_imputation --records_file Denoising_imputation_records.xls --data_class feature --data_name SHL --val_ratio 0.1 --test_ratio 0.2 --epochs 200 --lr 0.001 --optimizer RAdam --batch_size 729 --pos_encoding learnable --d_model 64 --num_workers 16 --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type clean

python src/main.py --output_dir experiments --comment "finetune for classification" --name ${dir_prefix}_SHL_dual_branch_finetune_clean --task dual_branch_classification --records_file Classification_records.xls --data_class trajectory_with_feature --data_name SHL --val_ratio 0.1 --test_ratio 0.2 --epochs 200 --lr 0.001 --optimizer RAdam --batch_size 450 --pos_encoding learnable --num_workers 16  --input_type clean --change_output --key_metric accuracy --denoising_model_hyperparams experiments/tmp/feature_model_hyperparams.json --imputation_model_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_denoising_branch experiments/tmp/feature_model_best.pth --load_imputation_branch experiments/tmp/trajectory_model_best.pth

#test
#python src/main.py --output_dir experiments --comment "finetune for classification" --name SHL_dual_branch_finetune_clean_test --task dual_branch_classification --records_file Classification_records.xls --data_class trajectory_with_feature --data_name SHL --val_ratio 0.1 --test_ratio 0.2 --epochs 200 --lr 0.001 --optimizer RAdam --batch_size 250 --pos_encoding learnable --num_workers 16 --key_metric accuracy --denoising_model_hyperparams experiments/tmp/feature_model_hyperparams.json --imputation_model_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_model experiments/SHL_dual_branch_finetune_clean_2022-05-07_23-23-33_8oQ/checkpoints/model_best.pth  --input_type noise --test_only testset


#---
python src/main.py --output_dir experiments --comment "pretraining through trj_imputation" --name ${dir_prefix}_geolife_trj_pretrained_clean --task denoising_imputation --records_file Denoising_imputation_records.xls --data_class trajectory --data_name geolife --val_ratio 0.1 --test_ratio 0.2 --epochs 10 --lr 0.001 --optimizer RAdam --batch_size 720 --pos_encoding learnable --num_workers 16 --d_model 64  --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type clean

python src/main.py --output_dir experiments --comment "pretraining through feat_denoising_imputation" --name ${dir_prefix}_geolife_feature_pretrained_clean --task denoising_imputation --records_file Denoising_imputation_records.xls --data_class feature --data_name geolife --val_ratio 0.1 --test_ratio 0.2 --epochs 200 --lr 0.001 --optimizer RAdam --batch_size 729 --pos_encoding learnable --d_model 64 --num_workers 16 --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type clean

python src/main.py --output_dir experiments --comment "finetune for classification" --name ${dir_prefix}_geolife_dual_branch_finetune_clean --task dual_branch_classification --records_file Classification_records.xls --data_class trajectory_with_feature --data_name geolife --val_ratio 0.1 --test_ratio 0.2 --epochs 150 --lr 0.001 --optimizer RAdam --batch_size 450 --pos_encoding learnable --num_workers 16  --input_type clean --change_output --key_metric accuracy --denoising_model_hyperparams experiments/tmp/feature_model_hyperparams.json --imputation_model_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_denoising_branch experiments/tmp/feature_model_best.pth --load_imputation_branch experiments/tmp/trajectory_model_best.pth

#test
#python src/main.py --output_dir experiments --comment "finetune for classification" --name geolife_dual_branch_finetune_clean_test --task dual_branch_classification --records_file Classification_records.xls --data_class trajectory_with_feature --data_name geolife --val_ratio 0.1 --test_ratio 0.2 --epochs 200 --lr 0.001 --optimizer RAdam --batch_size 250 --pos_encoding learnable --num_workers 16 --key_metric accuracy --denoising_model_hyperparams experiments/tmp/feature_model_hyperparams.json --imputation_model_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_model experiments/geolife_dual_branch_finetune_clean_2022-05-08_01-04-43_u9C/checkpoints/model_best.pth  --input_type noise --test_only testset
