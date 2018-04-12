#!/bin/bash
# You need to modify the dataset path. 
DATA_DIR="/vol/vssp/msos/audioset/packed_features"

# You can to modify to your own workspace. 
# WORKSPACE=`pwd`
WORKSPACE="/vol/vssp/msos/qk/workspaces/pub_audioset_classification"

# Train & predict. 
CUDA_VISIBLE_DEVICES=1 python keras/main.py train --data_dir=$DATA_DIR --workspace=$WORKSPACE --mini_data

# Compute averaged stats. 
python main.py get_avg_stats --cpickle_dir=$CPICKLE_DIR --workspace=$WORKSPACE

###
# If you extracted feature for new audio, you may do prediction using:
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python predict_new.py --workspace=$WORKSPACE --model_name=md20000_iters.p
