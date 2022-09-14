dataset='SHL' #
trj_pre_epoch=200
feat_pre_epoch=300
finetune_epoch=300
n_patience=20

trj_branch_methods=("denoising_pretrain" "denoising_imputation_pretrain" "denoising_pretrain" "denoising_imputation_pretrain")
trj_branch_models=("experiments/0820afternoon_repeat1_SHL_trj_denoising_pretrain_2022-08-20_15-55-18_E69" "experiments/0820afternoon_repeat1_SHL_trj_denoising_imputation_pretrain_2022-08-20_16-14-15_FpO" "experiments/0820afternoon_repeat1_SHL_trj_denoising_pretrain_2022-08-20_15-55-18_E69" "experiments/0820afternoon_repeat1_SHL_trj_denoising_imputation_pretrain_2022-08-20_16-14-15_FpO")
feature_branch_methods=("denoising_pretrain" "denoising_pretrain" "denoising_imputation_pretrain" "denoising_imputation_pretrain")
feature_branch_models=("experiments/0820afternoon_repeat2_SHL_feature_denoising_pretrain_2022-08-20_17-29-45_vIv" "experiments/0820afternoon_repeat2_SHL_feature_denoising_pretrain_2022-08-20_17-29-45_vIv" "experiments/0820afternoon_repeat2_SHL_feature_denoising_imputation_pretrain_2022-08-20_18-32-40_gzw" "experiments/0820afternoon_repeat2_SHL_feature_denoising_imputation_pretrain_2022-08-20_18-32-40_gzw")

input_type=("clean" "noise" "mix")

for i in $(seq 0 1 4); do
  dir_prefix="0820eve_repeat${i}"
  for j in $(seq 0 1 3); do
    tbm=${trj_branch_methods[${j}]}
    fbm=${feature_branch_methods[${j}]}
    tbmdl="${trj_branch_models[${j}]}/checkpoints/model_best.pth"
    fbmdl="${feature_branch_models[${j}]}/checkpoints/model_best.pth"

    python src/main.py --output_dir experiments --comment "finetune for classification" --name ${dir_prefix}_${dataset}_dual_branch_finetune_trj_${tbm}_feat_${fbm} --task dual_branch_classification --records_file ${dataset}_finetune_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${finetune_epoch} --lr 0.001 --optimizer RAdam --batch_size 680 --pos_encoding learnable --num_workers 16 --input_type mix --change_output --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_feature_branch ${fbmdl} --load_trajectory_branch ${tbmdl} --patience ${n_patience}
    sleep 5s
    #----------
    #test
    for k in $(seq 0 1 2); do
      python src/main.py --output_dir test_res --comment "test classification" --name ${dir_prefix}_${dataset}_trj_${tbm}_feat_${fbm}_classification_test --task dual_branch_classification --records_file ${dataset}_test_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --batch_size 600 --pos_encoding learnable --num_workers 16 --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_model experiments/tmp/trajectory_with_feature_model_best.pth --input_type ${input_type[${k}]} --test_only testset
      sleep 5s
    done
  done
done
