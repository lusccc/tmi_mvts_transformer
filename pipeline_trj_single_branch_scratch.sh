dataset='SHL'

epochs=250

#note using optimal hyperparameters
#features=("3" "4" "5" "8" "3,4" "3,5" "4,5" "3,4,5" "3,4,8" "3,5,8" "3,4,5,8" "0,1,2,3,4,5,6,7,8")
features=("3,4" "3,5" "4,5" "3,4,5" "3,4,8" "3,5,8" "3,4,5,8")

for j in $(seq 0 1 2); do
  prefix='831night'_repeat${j}
  for i in $(seq 0 1 6); do
    echo "${features[${i}]}"
    python src/main.py --output_dir experiments --comment "single branch classification from scratch" --name ${prefix}_${dataset}_feature"${features[${i}]}"_classification_scratch --task feature_branch_classification_from_scratch --records_file Classification_records.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${epochs} --lr 0.001 --optimizer RAdam --batch_size 3000 --pos_encoding learnable --d_model 128 --num_workers 16 --num_heads 8 --num_layers 3 --dim_feedforward 256 --input_type mix --key_metric accuracy --motion_features ${features[${i}]} --patience 20
    sleep 2s
  done
done

#python src/main.py --output_dir experiments --comment "single branch classification from scratch" --name octest_SHL_feature3,4,5,8_classification_scratch_2nd --task feature_branch_classification_from_scratch --records_file Classification_records.xls --data_class feature --data_name SHL --val_ratio 0.1 --test_ratio 0.2 --epochs 150 --lr 0.001 --optimizer RAdam --batch_size 700 --pos_encoding learnable --d_model 64 --num_workers 16 --num_heads 8 --num_layers 4 --dim_feedforward 256  --input_type mix --key_metric accuracy --motion_features 3,4,5,8

#for i in $(seq 0 1 10); do
#	echo i
#done
