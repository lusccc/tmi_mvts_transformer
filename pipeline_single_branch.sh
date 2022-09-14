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
for dataset in 'SHL' 'geolife'; do
  for branch in 'feature' 'trajectory'; do
    python src/main.py --output_dir experiments --comment "${branch} single branch pretrain" --name ${prefix}_${dataset}_${branch}_${pretrain_method} --task ${pretrain_method} --records_file ${dataset}_${branch}_single_pre_records.xls --data_class ${branch} --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${pre_epoch} --lr 0.001 --optimizer RAdam --batch_size 700 --d_model ${d_model} --num_heads ${num_heads} --num_layers ${num_layers} --dim_feedforward ${dim_feedforward} --input_type mix --patience ${n_patience} --pos_encoding learnable
    sleep 1s
    python src/main.py --output_dir experiments --comment "${branch} single branch fine tune classification" --name ${prefix}_${dataset}_${branch}_branch_classification --task ${branch}_branch_classification --records_file ${dataset}_${branch}_single_finetune_records.xls --data_class ${branch} --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${finetune_epoch} --lr 0.001 --optimizer RAdam --batch_size 700 --pos_encoding learnable --d_model ${d_model} --num_heads ${num_heads} --num_layers ${num_layers} --dim_feedforward ${dim_feedforward} --input_type mix --patience ${n_patience} --input_type mix --key_metric accuracy --${branch}_branch_hyperparams experiments/tmp/${branch}_model_hyperparams.json --load_model experiments/tmp/${branch}_model_best.pth --change_output
    sleep 1s
    #test
    for k in $(seq 0 1 2); do
      python src/main.py --output_dir test_res --comment "test classification" --name ${prefix}_${dataset}_${branch}_single_classification_test --task ${branch}_branch_classification --records_file ${dataset}_${branch}_single_test_records.xls --data_class ${branch} --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --batch_size 500 --num_workers 16 --key_metric accuracy --load_model experiments/tmp/${branch}_model_best.pth --input_type ${input_type[${k}]} --test_only testset --${branch}_branch_hyperparams experiments/tmp/${branch}_model_hyperparams.json
      sleep 1s
    done
  done
done
