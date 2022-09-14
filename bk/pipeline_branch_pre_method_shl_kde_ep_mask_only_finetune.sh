dataset='SHL_msk3' #
finetune_epoch=500
n_patience=80

trj_branch_methods=("denoising_pretrain" "denoising_imputation_pretrain" "denoising_pretrain" "denoising_imputation_pretrain")
feature_branch_methods=("denoising_pretrain" "denoising_pretrain" "denoising_imputation_pretrain" "denoising_imputation_pretrain")

input_type=("clean" "noise" "mix")

for i in $(seq 0 1 0); do
  dir_prefix="pc630afternoon_repeat${i}"
  for j in $(seq 0 1 3); do
    tbm=${trj_branch_methods[${j}]}
    fbm=${feature_branch_methods[${j}]}

    python src/main.py --output_dir experiments --comment "finetune for classification" --name ${dir_prefix}_${dataset}_dual_branch_finetune_trj_${tbm}_feat_${fbm} --task dual_branch_classification --records_file ${dataset}_finetune_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${finetune_epoch} --lr 0.001 --optimizer RAdam --batch_size 680 --pos_encoding learnable --num_workers 16 --input_type mix --change_output --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_feature_branch experiments/pre_models/feat_${fbm}.pth --load_trajectory_branch experiments/pre_models/trj_${tbm}.pth --patience ${n_patience}
    sleep 5s
    #----------
    #test
    for k in $(seq 0 1 2); do
      python src/main.py --output_dir test_res --comment "test classification" --name ${dir_prefix}_{dataset}_trj_${tbm}_feat_${fbm}_classification_test --task dual_branch_classification --records_file ${dataset}_test_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --batch_size 600 --pos_encoding learnable --num_workers 16 --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_model experiments/tmp/trajectory_with_feature_model_best.pth --input_type ${input_type[${k}]} --test_only testset
      sleep 5s
    done
  done
done
