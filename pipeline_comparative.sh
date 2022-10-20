#datasets=('SHL' 'geolife')
#methods=('cnn' 'lstm')

#for dataset in 'SHL' 'geolife'; do
for dataset in 'SHL'; do
  for method in 'cnn'; do
#  for method in 'cnn' 'lstm'; do
    n_patience=20
    input_type=("clean" "noise" "mix")
    for i in $(seq 0 1 0); do
      dir_prefix="0916_repeat${i}"
      python src/main.py --output_dir experiments --comment "${method} comparative exp" --name ${dir_prefix}_${dataset}_${method}_comparative_exp --task ${method}_classification --records_file ${dataset}_${method}_train_records.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --epochs 80 --lr 0.001 --optimizer RAdam --batch_size 700 --num_workers 16 --activation relu --input_type mix --patience ${n_patience}
      #test
      for k in $(seq 0 1 2); do
        python src/main.py --output_dir test_res --comment "test classification" --name ${dir_prefix}_${dataset}_${method}_classification_test --task ${method}_classification --records_file ${dataset}_${method}_test_records.xls --data_class feature --data_name ${dataset} --val_ratio 0.1 --test_ratio 0.2 --batch_size 700 --num_workers 16 --key_metric accuracy --load_model experiments/tmp/feature_model_best.pth --input_type ${input_type[${k}]} --test_only testset
        sleep 1s
      done
    done
  done
done
