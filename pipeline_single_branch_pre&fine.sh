dataset='geolife'
prefix='514midnight'
feature_branch_method='denoising_pretrain'
pre_epoch=10
finetune_epoch=100
python src/main.py --output_dir experiments --comment "feat ${feature_branch_method}" --name ${prefix}_${dataset}_feature_${feature_branch_method} --task ${feature_branch_method} --records_file feat_${feature_branch_method}.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${pre_epoch} --lr 0.001 --optimizer RAdam --batch_size 1200 --pos_encoding learnable --d_model 64 --num_workers 16 --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix
sleep 5s
python src/main.py --output_dir experiments --comment "single branch fine tune classification" --name ${prefix}_${dataset}_feature_branch_classification --task feature_branch_classification --records_file Classification_records.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${finetune_epoch} --lr 0.001 --optimizer RAdam --batch_size 1200 --pos_encoding learnable --d_model 64 --num_workers 16 --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix --key_metric accuracy  --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --load_model experiments/tmp/feature_model_best.pth --change_output
#tmp
#python src/main.py --output_dir experiments --comment "single branch fine tune classification" --name ${prefix}_${dataset}_feature_branch_classification --task feature_branch_classification --records_file Classification_records.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${finetune_epoch} --lr 0.001 --optimizer RAdam --batch_size 1000 --pos_encoding learnable --d_model 64 --num_workers 16 --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix --key_metric accuracy  --feature_branch_hyperparams experiments/tmp/feature_model_hyperparams.json --load_model experiments/514midnight_geolife_feature_denoising_pretrain_2022-05-14_00-51-27_t78/checkpoints/model_best.pth --change_output
#sleep 5s


#----
#trj_branch_method='denoising_imputation_pretrain'
#pre_epoch=100
#python src/main.py --output_dir experiments --comment "trj ${trj_branch_method}" --name ${prefix}_${dataset}_trj_${trj_branch_method} --task ${trj_branch_method} --records_file trj_${trj_branch_method}.xls --data_class trajectory --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${pre_epoch} --lr 0.001 --optimizer RAdam --batch_size 1000 --pos_encoding learnable --num_workers 16 --d_model 64  --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix

#sleep 5s
#python src/main.py --output_dir experiments --comment "single branch fine tune classification" --name ${prefix}_${dataset}_trj_branch_classification_scratch --task trajectory_branch_classification --records_file Classification_records.xls --data_class trajectory --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${finetune_epoch} --lr 0.001 --optimizer RAdam --batch_size 1100 --pos_encoding learnable --d_model 64 --num_workers 16 --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix  --key_metric accuracy  --trajectory_branch_hyperparams experiments/tmp/trajectory_model_hyperparams.json --load_model experiments/514midnight_geolife_trj_denoising_imputation_pretrain_2022-05-14_04-18-20_j25/checkpoints/model_best.pth --change_output