dataset='SHL' #
trj_pre_epoch=250
feat_pre_epoch=250
finetune_epoch=250
#trj_pre_epoch=10
#feat_pre_epoch=200
#finetune_epoch=150

trj_branch_methods=("denoising_pretrain" "denoising_imputation_pretrain" "denoising_pretrain" "denoising_imputation_pretrain")
feature_branch_methods=("denoising_pretrain" "denoising_pretrain" "denoising_imputation_pretrain" "denoising_imputation_pretrain")

for i in $(seq 0 1 0); do
  dir_prefix="610afternoon_repeat${i}"
  for j in $(seq 0 1 3); do
    tbm=${trj_branch_methods[${j}]}
    fbm=${feature_branch_methods[${j}]}

    python src/main.py --output_dir experiments --comment "trj ${tbm}" --name ${dir_prefix}_${dataset}_trj_${tbm} --task ${tbm} --records_file trj_${tbm}.xls --data_class trajectory --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${trj_pre_epoch} --lr 0.001 --optimizer RAdam --batch_size 1000 --pos_encoding learnable --num_workers 16 --d_model 64 --num_heads 8 --num_layers 4 --dim_feedforward 256 --input_type mix
    sleep 5s
    python src/main.py --output_dir experiments --comment "feat ${fbm}" --name ${dir_prefix}_${dataset}_feature_${fbm} --task ${fbm} --records_file feat_${fbm}.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${feat_pre_epoch} --lr 0.001 --optimizer RAdam --batch_size 1000 --pos_encoding learnable --d_model 64 --num_workers 16 --num_heads 8 --num_layers 4 --dim_feedforward 256 --input_type mix
    sleep 5s
    python src/main.py --output_dir experiments --comment "finetune for classification" --name ${dir_prefix}_${dataset}_dual_branch_finetune_trj_${tbm}_feat_${fbm} --task dual_branch_classification --records_file Classification_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${finetune_epoch} --lr 0.001 --optimizer RAdam --batch_size 550 --pos_encoding learnable --num_workers 16 --input_type mix --change_output --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_feature_branch experiments/tmp/feature_model_best.pth --load_trajectory_branch experiments/tmp/trajectory_model_best.pth
    sleep 5s
  done
done

