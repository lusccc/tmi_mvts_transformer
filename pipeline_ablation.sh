
dataset='geolife'
prefix='disable_mask'

# spatial mask
#python src/main.py --output_dir experiments --comment "pretraining through trj_denoising_imputation" --name ${dataset}_${prefix}_trj_pretrained --task denoising_imputation_pretrain --records_file Denoising_imputation_records.xls --data_class trajectory --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 10 --lr 0.001 --optimizer RAdam --batch_size 800 --pos_encoding learnable --num_workers 16 --d_model 64  --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix --disable_mask
#sleep 5s
#python src/main.py --output_dir experiments --comment "pretraining through feat_denoising_imputation" --name ${dataset}_${prefix}_feature_pretrained --task denoising_imputation_pretrain --records_file Denoising_imputation_records.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 10 --lr 0.001 --optimizer RAdam --batch_size 800 --pos_encoding learnable --d_model 64 --num_workers 16 --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix --disable_mask
#sleep 5s
#python src/main.py --output_dir experiments --comment "finetune for classification disable_mask" --name ${dataset}_${prefix}_dual_branch_finetune --task dual_branch_classification --records_file Classification_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 80 --lr 0.001 --optimizer RAdam --batch_size 450 --pos_encoding learnable --num_workers 16  --input_type mix --change_output --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_feature_branch experiments/tmp/feature_model_best.pth --load_trajectory_branch experiments/tmp/trajectory_model_best.pth --disable_mask --load_model experiments/geolife_disable_mask_dual_branch_finetune_2022-05-10_00-13-46_uB9/checkpoints/model_last.pth


prefix='scratch'
for i in $(seq 0 1 2 ); do
  python src/main.py --output_dir experiments --comment "dual branch classification from scratch" --name ${dataset}_${prefix}_dual_branch_classification --task dual_branch_classification_from_scratch --records_file Classification_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 250 --lr 0.001 --optimizer RAdam --batch_size 780 --pos_encoding learnable --d_model 64 --num_workers 16 --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json
  sleep 5s
done