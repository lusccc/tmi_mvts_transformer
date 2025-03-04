dataset='SHL' #
trj_pre_epoch=150
feat_pre_epoch=250
finetune_epoch=150
n_patience=20

#trj_branch_methods=("denoising_pretrain" "denoising_imputation_pretrain" "denoising_pretrain" "denoising_imputation_pretrain")
#feature_branch_methods=("denoising_pretrain" "denoising_pretrain" "denoising_imputation_pretrain" "denoising_imputation_pretrain")
trj_branch_methods=("denoising_imputation_pretrain")
feature_branch_methods=("denoising_imputation_pretrain")

input_type=("clean" "noise" "mix")

num_layers=3
num_heads=8
d_model=128
dim_feedforward=256
emb_size=64 # because default is 64

for i in $(seq 0 1 3); do
  dir_prefix="0908afternoon_repeat${i}"
#  for j in $(seq 0 1 3); do
  for j in $(seq 0 1 0); do
    tbm=${trj_branch_methods[${j}]}
    fbm=${feature_branch_methods[${j}]}

    python src/main.py --output_dir experiments --comment "trj ${tbm}" --name ${dir_prefix}_${dataset}_trj_${tbm} --task ${tbm} --records_file ${dataset}_trj_pre_records.xls --data_class trajectory --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${trj_pre_epoch} --lr 0.001 --optimizer RAdam --batch_size 850 --pos_encoding learnable --num_workers 16 --d_model ${d_model} --num_heads ${num_heads} --num_layers ${num_layers} --dim_feedforward ${dim_feedforward} --input_type mix --patience ${n_patience}
    sleep 1s
    python src/main.py --output_dir experiments --comment "feat ${fbm}" --name ${dir_prefix}_${dataset}_feature_${fbm} --task ${fbm} --records_file ${dataset}_fs_pre_records.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${feat_pre_epoch} --lr 0.001 --optimizer RAdam --batch_size 850 --pos_encoding learnable --d_model ${d_model} --num_workers 16 --num_heads ${num_heads} --num_layers ${num_layers} --dim_feedforward ${dim_feedforward} --input_type mix --patience ${n_patience}
    sleep 1s
    python src/main.py --output_dir experiments --comment "finetune for classification" --name ${dir_prefix}_${dataset}_dual_branch_finetune_trj_${tbm}_feat_${fbm} --task dual_branch_classification --records_file ${dataset}_finetune_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${finetune_epoch} --lr 0.001 --optimizer RAdam --batch_size 420 --pos_encoding learnable --num_workers 16 --input_type mix --change_output --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_feature_branch experiments/tmp/feature_model_best.pth --load_trajectory_branch experiments/tmp/trajectory_model_best.pth --patience 40
    sleep 1s
    #----------
    #test
    for k in $(seq 0 1 2); do
      python src/main.py --output_dir test_res --comment "test classification" --name ${dir_prefix}_${dataset}_trj_${tbm}_feat_${fbm}_classification_test --task dual_branch_classification --records_file ${dataset}_test_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --batch_size 300 --pos_encoding learnable --num_workers 16 --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_model experiments/tmp/trajectory_with_feature_model_best.pth --input_type ${input_type[${k}]} --test_only testset
      sleep 1s
    done
  done
done
