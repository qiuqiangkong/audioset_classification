import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import h5py
import argparse
import time
import logging
from sklearn import metrics
from utils import utilities, data_generator
import core

import keras
from keras.models import Model
from keras.layers import (Input, Dense, BatchNormalization, Dropout, Lambda,
                          Activation, Concatenate)
import keras.backend as K
from keras.optimizers import Adam

try:
    import cPickle
except BaseException:
    import _pickle as cPickle


def average_pooling(inputs, **kwargs):
    input = inputs[0]   # (batch_size, time_steps, freq_bins)
    return K.mean(input, axis=1)


def max_pooling(inputs, **kwargs):
    input = inputs[0]   # (batch_size, time_steps, freq_bins)
    return K.max(input, axis=1)


def attention_pooling(inputs, **kwargs):
    [out, att] = inputs

    epsilon = 1e-7
    att = K.clip(att, epsilon, 1. - epsilon)
    normalized_att = att / K.sum(att, axis=1)[:, None, :]

    return K.sum(out * normalized_att, axis=1)


def pooling_shape(input_shape):

    if isinstance(input_shape, list):
        (sample_num, time_steps, freq_bins) = input_shape[0]

    else:
        (sample_num, time_steps, freq_bins) = input_shape

    return (sample_num, freq_bins)


def train(args):
    
    model_type = args.model_type

    time_steps = 10
    freq_bins = 128
    classes_num = 527

    # Hyper parameters
    hidden_units = 1024
    drop_rate = 0.5
    batch_size = 500

    # Embedded layers
    input_layer = Input(shape=(time_steps, freq_bins))

    a1 = Dense(hidden_units)(input_layer)
    a1 = BatchNormalization()(a1)
    a1 = Activation('relu')(a1)
    a1 = Dropout(drop_rate)(a1)

    a2 = Dense(hidden_units)(a1)
    a2 = BatchNormalization()(a2)
    a2 = Activation('relu')(a2)
    a2 = Dropout(drop_rate)(a2)

    a3 = Dense(hidden_units)(a2)
    a3 = BatchNormalization()(a3)
    a3 = Activation('relu')(a3)
    a3 = Dropout(drop_rate)(a3)

    # Pooling layers
    if model_type == 'decision_level_max_pooling':
        '''Global max pooling.
        
        [1] Choi, Keunwoo, et al. "Automatic tagging using deep convolutional 
        neural networks." arXiv preprint arXiv:1606.00298 (2016).
        '''
        cla = Dense(classes_num, activation='sigmoid')(a3)
        output_layer = Lambda(max_pooling, output_shape=pooling_shape)([cla])

    elif model_type == 'decision_level_average_pooling':
        '''Global average pooling.
        
        [2] Lin, Min, et al. Qiang Chen, and Shuicheng Yan. "Network in 
        network." arXiv preprint arXiv:1312.4400 (2013).
        '''
        cla = Dense(classes_num, activation='sigmoid')(a3)
        output_layer = Lambda(
            average_pooling,
            output_shape=pooling_shape)(
            [cla])

    elif model_type == 'decision_level_single_attention':
        '''Decision level single attention pooling.

        [3] Kong, Qiuqiang, et al. "Audio Set classification with attention
        model: A probabilistic perspective." arXiv preprint arXiv:1711.00927
        (2017).
        '''
        cla = Dense(classes_num, activation='sigmoid')(a3)
        att = Dense(classes_num, activation='softmax')(a3)
        output_layer = Lambda(
            attention_pooling, output_shape=pooling_shape)([cla, att])

    elif model_type == 'decision_level_multi_attention':
        '''Decision level multi attention pooling.

        [4] Yu, Changsong, et al. "Multi-level Attention Model for Weakly
        Supervised Audio Classification." arXiv preprint arXiv:1803.02353
        (2018).
        '''
        cla1 = Dense(classes_num, activation='sigmoid')(a2)
        att1 = Dense(classes_num, activation='softmax')(a2)
        out1 = Lambda(
            attention_pooling, output_shape=pooling_shape)([cla1, att1])

        cla2 = Dense(classes_num, activation='sigmoid')(a3)
        att2 = Dense(classes_num, activation='softmax')(a3)
        out2 = Lambda(
            attention_pooling, output_shape=pooling_shape)([cla2, att2])

        b1 = Concatenate(axis=-1)([out1, out2])
        b1 = Dense(classes_num)(b1)
        output_layer = Activation('sigmoid')(b1)

    elif model_type == 'feature_level_attention':
        '''Feature level attention.

        [1] To be appear.
        '''
        cla = Dense(hidden_units, activation='linear')(a3)
        att = Dense(hidden_units, activation='sigmoid')(a3)
        b1 = Lambda(
            attention_pooling, output_shape=pooling_shape)([cla, att])

        b1 = BatchNormalization()(b1)
        b1 = Activation(activation='relu')(b1)
        b1 = Dropout(drop_rate)(b1)

        output_layer = Dense(classes_num, activation='sigmoid')(b1)

    else:
        raise Exception("Incorrect model_type!")

    # Build model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    args.model = model
    args.batch_size = batch_size

    # Train
    core.train(args)


# Main
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--data_dir', type=str, required=True)

    parser.add_argument('--workspace', type=str, required=True)

    parser.add_argument('--mini_data', action='store_true',
                        default=False)

    parser.add_argument('--balance_type', type=str,
                        default='balance_in_batch',
                        choices=['no_balance', 'balance_in_batch'])

    parser.add_argument('--model_type', type=str, required=True,
                        choices=['decision_level_max_pooling', 
                                 'decision_level_average_pooling', 
                                 'decision_level_single_attention',
                                 'decision_level_multi_attention',
                                 'feature_level_attention'])

    parser.add_argument('--learning_rate', type=float, default=1e-3)

    subparsers = parser.add_subparsers(dest='mode')
    parser_train = subparsers.add_parser('train')
    parser_get_avg_stats = subparsers.add_parser('get_avg_stats')

    args = parser.parse_args()

    args.filename = utilities.get_filename(__file__)

    # Logs
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    utilities.create_folder(logs_dir)
    logging = utilities.create_logging(logs_dir, filemode='w')

    logging.info(os.path.abspath(__file__))
    logging.info(args)

    if args.mode == "train":
        train(args)

    elif args.mode == 'get_avg_stats':
        args.bgn_iteration = 10000
        args.fin_iteration = 50001
        args.interval_iteration = 5000
        utilities.get_avg_stats(args)

    else:
        raise Exception("Error!")
