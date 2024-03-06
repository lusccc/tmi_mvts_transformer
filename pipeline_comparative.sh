#batch_size=350
#for dataset in  'SHL' 'geolife'; do
#  for method in 'feature_branch_classification_from_scratch'; do
#    n_patience=20
#    epochs=120
#    input_type=("clean" "noise" "mix")
#    for i in $(seq 0 1 0); do
#      prefix="0218_LN_adam_posfixed_512,8,6"
#      prefix2="0218"
#      python main.py --output_dir experiments --comment "${method} comparative exp" --name ${prefix}_${dataset}_${method}_comparative_exp --task ${method} --records_file ${prefix2}_${dataset}_${method}_train_records.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${epochs} --lr 0.001 --optimizer Adam --batch_size ${batch_size} --num_workers 24 --input_type mix --patience ${n_patience} --normalization_layer LayerNorm --d_model 512  --num_heads 8 --num_layers 6 --dim_feedforward 256
#      #test
#      for k in $(seq 0 1 2); do
#        python main.py --output_dir test_res --comment "test classification" --name ${prefix}_${dataset}_${method}_classification_test --task ${method} --records_file ${prefix2}_${dataset}_${method}_test_records.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --batch_size 60 --num_workers 24 --key_metric accuracy --load_model experiments/tmp/feature_model_best.pth --input_type ${input_type[${k}]} --test_only testset --normalization_layer LayerNorm --d_model 512  --num_heads 8 --num_layers 6 --dim_feedforward 256
#        sleep 1s
#      done
#    done
#  done
#done



batch_size=1500
for dataset in 'geolife'; do
  for method in 'cnn_classification'; do
#  for method in 'feature_branch_classification_from_scratch'; do
    n_patience=5
    epochs=25
    input_type=("clean" "noise" "mix")
    for i in $(seq 0 1 10); do
      prefix="0302"
      prefix2="0302"
      python main.py --output_dir experiments --comment "${method} comparative exp" --name ${prefix}_${dataset}_${method}_comparative_exp --task ${method} --records_file ${prefix2}_${dataset}_${method}_train_records.xlsx --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${epochs} --lr 0.001 --optimizer Adam --batch_size ${batch_size} --num_workers 24 --input_type mix --patience ${n_patience}
      #test
      for k in $(seq 0 1 2); do
        python main.py --output_dir test_res --comment "test classification" --name ${prefix}_${dataset}_${method}_classification_test --task ${method} --records_file ${prefix2}_${dataset}_${method}_test_records.xlsx --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --batch_size ${batch_size} --num_workers 24 --key_metric accuracy --load_model experiments/tmp/feature_model_best.pth --input_type ${input_type[${k}]} --test_only testset
        sleep 1s
      done
    done
  done
done

#for dataset in 'SHL' 'geolife'; do
#  for method in 'lstm_classification'; do
##  for method in 'feature_branch_classification_from_scratch'; do
#    n_patience=30
#    epochs=100
#    input_type=("clean" "noise" "mix")
#    for i in $(seq 0 1 0); do
#      prefix="0218"
#      prefix2="0218"
#      python main.py --output_dir experiments --comment "${method} comparative exp" --name ${prefix}_${dataset}_${method}_comparative_exp --task ${method} --records_file ${prefix2}_${dataset}_${method}_train_records.xlsx --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs ${epochs} --lr 0.001 --optimizer Adam --batch_size ${batch_size} --num_workers 24 --activation relu --input_type mix --patience ${n_patience}
#      #test
#      for k in $(seq 0 1 2); do
#        python main.py --output_dir test_res --comment "test classification" --name ${prefix}_${dataset}_${method}_classification_test --task ${method} --records_file ${prefix2}_${dataset}_${method}_test_records.xlsx --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --batch_size ${batch_size} --num_workers 24 --key_metric accuracy --load_model experiments/tmp/feature_model_best.pth --input_type ${input_type[${k}]} --test_only testset
#        sleep 1s
#      done
#    done
#  done
#done
