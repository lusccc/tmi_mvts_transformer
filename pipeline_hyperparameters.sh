dataset='SHL'
trj_pre_epoch=150
feat_pre_epoch=200
finetune_epoch=250
n_patience=20

#num_layers num_heads d_model  dim_feedforward    emb_size
#2            8         64      256                  64
#2            16        64      256                  64
num_layers=3
num_heads=8
d_model=128
dim_feedforward=256
emb_size=64

for i in $(seq 0 1 0); do
  dir_prefix="0826night_repeat${i}"
  tbm="denoising_imputation_pretrain"
  fbm="denoising_imputation_pretrain"
  python src/main.py --output_dir experiments --comment "trj ${tbm}" --name ${dir_prefix}_${dataset}_trj_${tbm} --task ${tbm} --records_file ${dataset}_trj_pre_records.xls --data_class trajectory --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${trj_pre_epoch} --lr 0.001 --optimizer RAdam --batch_size 900 --pos_encoding learnable --num_workers 16 --d_model ${d_model} --num_heads ${num_heads} --num_layers ${num_layers} --dim_feedforward ${dim_feedforward} --input_type mix --patience ${n_patience}
  sleep 5s
  python src/main.py --output_dir experiments --comment "feat ${fbm}" --name ${dir_prefix}_${dataset}_feature_${fbm} --task ${fbm} --records_file ${dataset}_fs_pre_records.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${feat_pre_epoch} --lr 0.001 --optimizer RAdam --batch_size 900 --pos_encoding learnable --d_model ${d_model} --num_workers 16 --num_heads ${num_heads} --num_layers ${num_layers} --dim_feedforward ${dim_feedforward} --input_type mix --patience ${n_patience}
  sleep 5s
  python src/main.py --output_dir experiments --comment "finetune for classification" --name ${dir_prefix}_${dataset}_dual_branch_finetune_trj_${tbm}_feat_${fbm} --task dual_branch_classification --records_file ${dataset}_finetune_records.xls --data_class trajectory_with_feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${finetune_epoch} --lr 0.001 --optimizer RAdam --batch_size 450 --pos_encoding learnable --num_workers 16 --input_type mix --change_output --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_feature_branch experiments/tmp/feature_model_best.pth --load_trajectory_branch experiments/tmp/trajectory_model_best.pth --patience 40 --emb_size ${emb_size}
  sleep 5s
done