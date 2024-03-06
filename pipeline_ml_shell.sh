python  main.py --output_dir experiments --comment ml --name geo_ml --task ml_classification --data_class feature --data_name geolife --val_ratio 0.1 --test_ratio 0.2 --batch_size 700 --pos_encoding learnable --input_type mix --motion_features "0,2,3,4,7,8" --num_workers 24

python  main.py --output_dir experiments --comment ml --name geo_ml --task ml_classification --data_class feature --data_name geolife --val_ratio 0.1 --test_ratio 0.2 --batch_size 700 --pos_encoding learnable --input_type noise --motion_features "0,2,3,4,7,8" --num_workers 24 --test_only testset



python  main.py --output_dir experiments --comment ml --name shl_ml --task ml_classification --data_class feature --data_name SHL --val_ratio 0.1 --test_ratio 0.2 --batch_size 700 --pos_encoding learnable --input_type mix --motion_features "0,2,3,4,7,8" --num_workers 24

python  main.py --output_dir experiments --comment ml --name shl_ml --task ml_classification --data_class feature --data_name SHL --val_ratio 0.1 --test_ratio 0.2 --batch_size 700 --pos_encoding learnable --input_type noise --motion_features "0,2,3,4,7,8" --num_workers 24 --test_only testset


