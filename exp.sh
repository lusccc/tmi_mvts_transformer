#++++++++++++++++++++++++++++++++
#++++++++++++++SHL+++++++++++
#++++++++++++++++++++++++++++++++
python src/data_preprocess/trajectory_feature_calcuation.py --trjs_path ./data/SHL_extracted/trjs.npy --labels_path ./data/SHL_extracted/labels.npy --n_class 5 --save_dir ./data/SHL_features/

python src/main.py --output_dir experiments --comment "pretraining through imputation" --name SHL_pretrained --records_file Imputation_records.xls --data_class SHL --val_ratio 0.2 --epochs 40 --lr 0.001 --optimizer RAdam --batch_size 1200 --pos_encoding learnable --d_model 128

python src/main.py --output_dir experiments --comment "finetune for classification" --name SHL_finetuned --records_file Classification_records.xls --data_class SHL --val_ratio 0.1 --test_ratio 0.2 --epochs 100 --lr 0.001 --optimizer RAdam --batch_size 1200 --pos_encoding learnable --d_model 128 --load_model ./experiments/SHL_pretrained_2021-10-03_13-05-13_Q2l/checkpoints/model_best.pth --task classification --change_output --key_metric accuracy

python src/main.py --output_dir experiments --comment "finetune for classification" --name SHL_finetuned --records_file Classification_records.xls --data_class SHL --val_ratio 0.1 --test_ratio 0.2 --epochs 100 --lr 0.001 --optimizer RAdam --batch_size 1200 --pos_encoding learnable --d_model 128 --load_model experiments/SHL_pretrained_2021-10-08_00-40-16_Ylz/checkpoints/model_best.pth --task classification --change_output --key_metric accuracy

#----------------10.21------------------
python src/main.py --output_dir experiments --comment "pretraining through feat_imputation" --name SHL_feature_pretrained --task denoising --records_file Imputation_records.xls --data_class SHL_feature --val_ratio 0.2 --epochs 40 --lr 0.001 --optimizer RAdam --batch_size 1200 --pos_encoding learnable --d_model 128

python src/main.py --output_dir experiments --comment "pretraining through trj_imputation" --name SHL_trj_pretrained --records_file Imputation_records.xls --data_class SHL_trajectory --val_ratio 0.2 --epochs 40 --lr 0.001 --optimizer RAdam --batch_size 1200 --pos_encoding learnable --d_model 128 --normalization per_sample_std

#----------------10.25------------------
python src/main.py --output_dir experiments --comment "pretraining through feat_denoising" --name SHL_feature_pretrained --task denoising --records_file Denoising_records.xls --data_class SHL_feature --val_ratio 0.2 --epochs 40 --lr 0.001 --optimizer RAdam --batch_size 500 --pos_encoding learnable --d_model 128

python src/main.py --output_dir experiments --comment "pretraining through trj_imputation" --name SHL_trj_pretrained --records_file Imputation_records.xls --data_class SHL_trajectory --val_ratio 0.2 --epochs 40 --lr 0.001 --optimizer RAdam --batch_size 600 --pos_encoding learnable --d_model 128 --normalization per_sample_minmax

#----------------10.28------------------
python src/main.py --output_dir experiments --comment "pretraining through feat_denoising" --name SHL_feature_pretrained --task denoising --records_file Denoising_records.xls --data_class feature --data_name SHL --val_ratio 0.2 --epochs 200 --lr 0.001 --optimizer RAdam --batch_size 500 --pos_encoding learnable --d_model 128

python src/main.py --output_dir experiments --comment "pretraining through trj_imputation" --name SHL_trj_pretrained --task imputation --records_file Imputation_records.xls --data_class trajectory --data_name SHL --val_ratio 0.2 --epochs 200 --lr 0.001 --optimizer RAdam --batch_size 520 --pos_encoding learnable --d_model 128 --normalization per_sample_minmax

python src/main.py --output_dir experiments --comment "dual branch finetune for classification" --name SHL_dual_branch_finetuned --records_file Classification_records.xls  --data_class trajectory_with_feature --data_name SHL --val_ratio 0.1 --test_ratio 0.2 --epochs 300 --lr 0.001 --optimizer RAdam --batch_size 550 --pos_encoding learnable --d_model 128 --task dual_branch_classification --change_output --key_metric accuracy --imputation_model_hyperparams experiments/SHL_trj_pretrained_2021-10-30_21-54-44_HL1/imputation_model_hyperparams.json --denoising_model_hyperparams experiments/SHL_feature_pretrained_2021-10-30_21-51-15_Gfe/denoising_model_hyperparams.json --load_imputation_branch experiments/SHL_trj_pretrained_2021-10-30_21-54-44_HL1/checkpoints/model_best.pth --load_denoising_branch experiments/SHL_feature_pretrained_2021-10-30_21-51-15_Gfe/checkpoints/model_best.pth

#----------------10.30------------------
python src/main.py --output_dir experiments --comment "dual branch finetune for classification" --name SHL_dual_branch_finetuned --records_file Classification_records.xls  --data_class trajectory_with_feature --data_name SHL --val_ratio 0.1 --test_ratio 0.2 --epochs 300 --lr 0.001 --optimizer RAdam --batch_size 550 --pos_encoding learnable --d_model 128 --task dual_branch_classification --change_output --key_metric accuracy --imputation_model_hyperparams experiments/SHL_trj_pretrained_2021-10-28_17-01-13_maQ/imputation_model_hyperparams.json --denoising_model_hyperparams experiments/SHL_feature_pretrained_2021-10-28_16-48-00_cHt/denoising_model_hyperparams.json --load_imputation_branch experiments/SHL_trj_pretrained_2021-10-28_17-01-13_maQ/checkpoints/model_best.pth --load_denoising_branch experiments/SHL_feature_pretrained_2021-10-28_16-48-00_cHt/checkpoints/model_best.pth



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++GEOLIFE+++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


python src/data_preprocess/trajectory_feature_calculation.py  --trjs_path ./data/geolife_extracted/trjs.npy --labels_path ./data/geolife_extracted/labels.npy --n_class 5 --save_dir ./data/geolife_features/

python src/main.py --output_dir experiments --comment "pretraining through imputation" --name geolife_pretrained --records_file Imputation_records.xls --data_class geolife --val_ratio 0.2 --epochs 40 --lr 0.001 --optimizer RAdam --batch_size 1200 --pos_encoding learnable --d_model 128 --num_workers 16

python src/main.py --output_dir experiments --comment "finetune for classification" --name geolife_finetuned --records_file Classification_records.xls --data_class geolife --val_ratio 0.1 --test_ratio 0.2 --epochs 40 --lr 0.001 --optimizer RAdam --batch_size 1200 --pos_encoding learnable --d_model 128 --load_model experiments/geolife_pretrained_2021-10-08_13-36-50_BHd/checkpoints/model_best.pth --task classification --change_output --key_metric accuracy --num_workers 16


python src/main.py --output_dir experiments --comment "finetune for classification" --name geolife_finetuned --records_file Classification_records.xls --data_class geolife --val_ratio 0.1 --test_ratio 0.2 --epochs 200 --lr 0.001 --optimizer RAdam --batch_size 1200 --pos_encoding learnable --d_model 128 --load_model experiments/geolife_finetuned_2021-10-08_14-19-17_DZ2/checkpoints/model_best.pth --task classification --change_output --key_metric accuracy --num_workers 16


#----------------10.30------------------
python src/data_preprocess/trajectory_feature_calculation.py  --trjs_path ./data/geolife_extracted/trjs.npy --labels_path ./data/geolife_extracted/labels.npy --n_class 5 --save_dir ./data/geolife_features/

python src/main.py --output_dir experiments --comment "pretraining through feat_denoising" --name geolife_feature_pretrained --task denoising --records_file Denoising_records.xls --data_class feature --data_name geolife --val_ratio 0.2 --epochs 200 --lr 0.001 --optimizer RAdam --batch_size 510 --pos_encoding learnable --d_model 128 --num_workers 8

python src/main.py --output_dir experiments --comment "pretraining through trj_imputation" --name geolife_trj_pretrained --task imputation --records_file Imputation_records.xls --data_class trajectory --data_name geolife --val_ratio 0.2 --epochs 200 --lr 0.001 --optimizer RAdam --batch_size 510 --pos_encoding learnable --d_model 128 --normalization per_sample_minmax --num_workers 8

python src/main.py --output_dir experiments --comment "dual branch finetune for classification" --name geolife_dual_branch_finetuned --records_file Classification_records.xls  --data_class trajectory_with_feature --data_name geolife --val_ratio 0.1 --test_ratio 0.2 --epochs 300 --lr 0.001 --optimizer RAdam --batch_size 550 --pos_encoding learnable --d_model 128 --task dual_branch_classification --change_output --key_metric accuracy --imputation_model_hyperparams experiments/geolife_trj_pretrained_2021-10-31_00-29-18_HRj/imputation_model_hyperparams.json --denoising_model_hyperparams experiments/geolife_feature_pretrained_2021-10-31_00-29-09_blk/denoising_model_hyperparams.json --load_imputation_branch experiments/geolife_trj_pretrained_2021-10-31_00-29-18_HRj/checkpoints/model_best.pth --load_denoising_branch experiments/geolife_feature_pretrained_2021-10-31_00-29-09_blk/checkpoints/model_best.pth --num_workers 8