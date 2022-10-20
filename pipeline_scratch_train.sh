#!/bin/bash -e

input_type=("clean" "noise" "mix")
finetune_epoch=300
n_patience=20

num_layers=3
num_heads=8
d_model=128
dim_feedforward=256

tbm="denoising_imputation_pretrain"
fbm="denoising_imputation_pretrain"
n_patience=20
dir_prefix='0914morning'
for dataset in 'SHL' 'geolife'; do
  python src/main.py --output_dir experiments --comment "finetune for classification" --name ${dir_prefix}_${dataset}_dual_branch_finetune_trj_${tbm}_feat_${fbm} --task dual_branch_classification_from_scratch --records_file ${dataset}_finetune_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${finetune_epoch} --lr 0.001 --optimizer RAdam --batch_size 380 --pos_encoding learnable --num_workers 16 --input_type mix --change_output --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --patience 40
  sleep 1s
  #----------
  #test
  for k in $(seq 0 1 2); do
    python src/main.py --output_dir test_res --comment "test classification" --name ${dir_prefix}_${dataset}_trj_${tbm}_feat_${fbm}_classification_test --task dual_branch_classification --records_file ${dataset}_test_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --batch_size 300 --pos_encoding learnable --num_workers 16 --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_model experiments/tmp/trajectory_with_feature_model_best.pth --input_type ${input_type[${k}]} --test_only testset
    sleep 1s
  done
done
