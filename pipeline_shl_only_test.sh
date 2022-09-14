dataset='SHL'
tbm='denoising_imputation_pretrain'
fbm='denoising_imputation_pretrain'
finetune_epoch=250
input_type=("clean" "noise" "mix")
dir_prefix="0906afternoon_test"
for k in $(seq 0 1 2); do
    python src/main.py --output_dir test_res --comment "test classification" --name ${dir_prefix}_${dataset}_trj_${tbm}_feat_${fbm}_classification_test --task dual_branch_classification --records_file ${dataset}_test_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --batch_size 300 --pos_encoding learnable --num_workers 16 --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_model history_exps/exp0831/0901morning_repeat3_SHL_dual_branch_finetune_trj_denoising_imputation_pretrain_feat_denoising_imputation_pretrain_2022-09-01_09-19-50_Ghd/checkpoints/model_best.pth --input_type ${input_type[${k}]} --test_only testset
    sleep 1s
  done
