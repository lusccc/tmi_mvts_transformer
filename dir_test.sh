dirs=("619afternoon_repeat0_geolife_dual_branch_finetune_trj_denoising_pretrain_feat_denoising_pretrain_2022-06-19_17-53-23_hby" "619afternoon_repeat0_geolife_dual_branch_finetune_trj_denoising_imputation_pretrain_feat_denoising_pretrain_2022-06-19_18-34-42_sXc" "619afternoon_repeat0_geolife_dual_branch_finetune_trj_denoising_pretrain_feat_denoising_imputation_pretrain_2022-06-19_19-19-40_1xl" "619afternoon_repeat0_geolife_dual_branch_finetune_trj_denoising_imputation_pretrain_feat_denoising_imputation_pretrain_2022-06-19_19-56-44_Ad7")
dataset='geolife'
trj_branch_methods=("denoising_pretrain" "denoising_imputation_pretrain" "denoising_pretrain" "denoising_imputation_pretrain")
feature_branch_methods=("denoising_pretrain" "denoising_pretrain" "denoising_imputation_pretrain" "denoising_imputation_pretrain")
input_type=("clean" "noise" "mix")
for j in $(seq 0 1 3); do
    tbm=${trj_branch_methods[${j}]}
    fbm=${feature_branch_methods[${j}]}
    for k in $(seq 0 1 2); do
      python src/main.py --output_dir test_res --comment "test classification" --name ${dataset}_trj_${tbm}_feat_${fbm}_classification_test --task dual_branch_classification --records_file ${dataset}_test_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --batch_size 600 --pos_encoding learnable --num_workers 16 --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_model experiments/${dirs[${j}]}/checkpoints/model_best.pth --input_type ${input_type[${k}]} --test_only testset

    done
done
