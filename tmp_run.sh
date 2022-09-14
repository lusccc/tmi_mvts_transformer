input_type=("clean" "noise" "mix")
pre_epoch=150
finetune_epoch=150

num_layers=3
num_heads=8
d_model=128
dim_feedforward=256

pretrain_method="denoising_imputation_pretrain"
n_patience=20
prefix='0910dawn'
dataset=SHL
branch=trajectory
branch=feature
for k in $(seq 0 1 2); do
  python src/main.py --output_dir test_res --comment "test classification" --name ${prefix}_${dataset}_${branch}_single_classification_test --task ${branch}_branch_classification --records_file ${dataset}_${branch}_single_test_records.xls --data_class ${branch} --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --batch_size 500 --num_workers 16 --key_metric accuracy --load_model experiments/tmp/${branch}_model_best.pth --input_type clean --test_only testset --${branch}_branch_hyperparams experiments/tmp/${branch}_model_hyperparams.json
  sleep 1s
done

python src/main.py --output_dir experiments --comment "trajectory single branch fine tune classification" --name 0910dawn_SHL_trajectory_branch_classification --task trajectory_branch_classification --records_file SHL_trajectory_single_finetune_records.xls --data_class trajectory --data_name SHL --val_ratio 0.1 --test_ratio 0.2 --epochs 150 --lr 0.001 --optimizer RAdam --batch_size 700 --pos_encoding learnable --d_model 128 --num_heads 8 --num_layers 3 --dim_feedforward 256 --input_type mix --patience 20 --input_type mix --key_metric accuracy --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --load_model experiments/tmp/trajectory_model_best.pth --change_output

