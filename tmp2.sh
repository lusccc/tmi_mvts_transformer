#dataset='geolife'
#suffix='LN'
#python main.py --output_dir experiments --comment "pretraining through trj_denoising_imputation" --name ${dataset}_trj_pretrained_denoi_imp_${suffix} --task denoising_imputation_pretrain --records_file Denoising_imputation_records.xls --data_class trajectory --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 50 --lr 0.001 --optimizer RAdam --batch_size 1000 --pos_encoding learnable --num_workers 24 --d_model 128  --num_heads 8 --num_layers 3 --dim_feedforward 256  --input_type mix --normalization_layer LayerNorm
#sleep 5s
#python main.py --output_dir experiments --comment "pretraining through feat_denoising_imputation" --name ${dataset}_feature_pretrained_denoi_imp_${suffix} --task denoising_imputation_pretrain --records_file Denoising_imputation_records.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 300 --lr 0.001 --optimizer RAdam --batch_size 1000 --pos_encoding learnable --d_model 64 --num_workers 24 --d_model 128  --num_heads 8 --num_layers 3 --dim_feedforward 256  --input_type mix --normalization_layer LayerNorm
#sleep 5s
#python main.py --output_dir experiments --comment "finetune for classification" --name ${dataset}_dual_branch_finetune_trj_denoi_imp_mf_denoi_imp_${suffix} --task dual_branch_classification --records_file Classification_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 300 --lr 0.001 --optimizer RAdam --batch_size 550 --pos_encoding learnable --num_workers 24  --input_type mix --change_output --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_feature_branch experiments/tmp/feature_model_best.pth --load_trajectory_branch experiments/tmp/trajectory_model_best.pth --normalization_layer LayerNorm
#sleep 5s


suffix='LN'
dataset='geolife'
batch_size=100
method='dual_branch_classification'
input_type=("clean" "noise" "mix")
for k in $(seq 0 1 2); do
  python main.py --output_dir test_res --comment "test classification" --name ${dataset}_dual_branch_finetune_trj_denoi_imp_mf_denoi_imp_${suffix} --task ${method} --records_file ${dataset}_${method}_test_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --batch_size ${batch_size} --num_workers 24 --key_metric accuracy --load_model experiments/geolife_dual_branch_finetune_trj_denoi_imp_mf_denoi_imp_LN_2024-02-17_17-23-09_XTX/checkpoints/model_best.pth --input_type ${input_type[${k}]} --test_only testset --normalization_layer LayerNorm --d_model 512  --num_heads 8 --num_layers 6 --dim_feedforward 256 --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json
  sleep 1s
done