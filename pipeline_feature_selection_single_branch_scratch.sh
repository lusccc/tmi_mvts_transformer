dataset='SHL'
epochs=250
prefix='0206'

python main.py --output_dir experiments --comment "single_branch_classification_from_scratch" --name ${prefix}_${dataset}_feat_classification_scratch --task feature_branch_classification_from_scratch --records_file Classification_records.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${epochs} --lr 0.001 --optimizer RAdam --batch_size 1500 --pos_encoding fixed --d_model 128 --num_workers 24 --num_heads 8 --num_layers 3 --dim_feedforward 256 --input_type mix --key_metric accuracy  --patience 20

#echo "python main.py --output_dir experiments --comment "single_branch_classification_from_scratch" --name ${prefix}_${dataset}_trj_classification_scratch --task trajectory_branch_classification_from_scratch --records_file Classification_records.xls --data_class trajectory --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${epochs} --lr 0.001 --optimizer RAdam --batch_size 1500 --pos_encoding fixed --d_model 128 --num_workers 24 --num_heads 8 --num_layers 3 --dim_feedforward 256 --input_type mix --key_metric accuracy  --patience 20"



#for j in $(seq 0 1 2); do
#  prefix='0206'_repeat${j}
#  for i in $(seq 0 1 6); do
#    python src/main.py --output_dir experiments --comment "single branch classification from scratch" --name ${prefix}_${dataset}_trj_classification_scratch --task trajectory_branch_classification_from_scratch --records_file Classification_records.xls --data_class trajectory --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${epochs} --lr 0.001 --optimizer RAdam --batch_size 3000 --pos_encoding fixed --d_model 128 --num_workers 16 --num_heads 8 --num_layers 3 --dim_feedforward 256 --input_type mix --key_metric accuracy  --patience 20
#    sleep 2s
#  done
#done

