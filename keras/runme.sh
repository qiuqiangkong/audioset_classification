CUDA_VISIBLE_DEVICES=1 python keras/main.py --data_dir=$DATA_DIR --workspace=$WORKSPACE --mini_data --balance_type=balance_in_batch --model_type=feature_level_attention train

