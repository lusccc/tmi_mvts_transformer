dataset=SHL_msk3

python src/main.py --output_dir experiments --comment "trj deno" --name ${dataset}_trj_deno --task denoising_pretrain --records_file ${dataset}_trj_pre_records.xls --data_class trajectory --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 500 --lr 0.001 --optimizer RAdam --batch_size 1300 --pos_encoding learnable --num_workers 16 --d_model 64 --num_heads 8 --num_layers 4 --dim_feedforward 256 --input_type mix --patience 60
    sleep 5s


python src/main.py --output_dir experiments --comment "trj deno_imp" --name ${dataset}_trj_deno_imp --task denoising_imputation_pretrain --records_file ${dataset}_trj_pre_records.xls --data_class trajectory --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 500 --lr 0.001 --optimizer RAdam --batch_size 1300 --pos_encoding learnable --num_workers 16 --d_model 64 --num_heads 8 --num_layers 4 --dim_feedforward 256 --input_type mix --patience 60

python src/main.py --output_dir experiments --comment "feat deno" --name ${dataset}_feature_deno --task denoising_pretrain --records_file ${dataset}_fs_pre_records.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 1300 --lr 0.001 --optimizer RAdam --batch_size 820 --pos_encoding learnable --d_model 64 --num_workers 16 --num_heads 8 --num_layers 4 --dim_feedforward 256 --input_type mix --patience 60

python src/main.py --output_dir experiments --comment "feat deno_imp" --name ${dataset}_feature_deno_imp --task denoising_imputation_pretrain --records_file ${dataset}_fs_pre_records.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 1300 --lr 0.001 --optimizer RAdam --batch_size 820 --pos_encoding learnable --d_model 64 --num_workers 16 --num_heads 8 --num_layers 4 --dim_feedforward 256 --input_type mix --patience 60