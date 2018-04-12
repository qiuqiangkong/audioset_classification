import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import h5py
import argparse
import time
import math
import logging
from sklearn import metrics
from utils import utilities, data_generator
import core

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

try:
    import cPickle
except BaseException:
    import _pickle as cPickle


def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.weight.data.fill_(1.)


class Attention(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(Attention, self).__init__()

        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att,)
        init_layer(self.cla)

    def activate(self, x, activation):

        if activation == 'linear':
            return x

        elif activation == 'relu':
            return F.relu(x)

        elif activation == 'sigmoid':
            return F.sigmoid(x)

        elif activation == 'softmax':
            return F.softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        att = self.att(x)
        att = self.activate(att, self.att_activation)

        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)

        att = att[:, :, :, 0]   # (samples_num, classes_num, time_steps)
        cla = cla[:, :, :, 0]   # (samples_num, classes_num, time_steps)

        epsilon = 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)

        return x


class EmbeddingLayers(nn.Module):

    def __init__(self, freq_bins, hidden_units, drop_rate):
        super(EmbeddingLayers, self).__init__()

        self.freq_bins = freq_bins
        self.hidden_units = hidden_units
        self.drop_rate = drop_rate

        self.conv1 = nn.Conv2d(
            in_channels=freq_bins, out_channels=hidden_units,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.conv2 = nn.Conv2d(
            in_channels=hidden_units, out_channels=hidden_units,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.conv3 = nn.Conv2d(
            in_channels=hidden_units, out_channels=hidden_units,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.bn0 = nn.BatchNorm2d(freq_bins)
        self.bn1 = nn.BatchNorm2d(hidden_units)
        self.bn2 = nn.BatchNorm2d(hidden_units)
        self.bn3 = nn.BatchNorm2d(hidden_units)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)

        init_bn(self.bn0)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)

    def forward(self, input, return_layers=False):
        """input: (samples_num, time_steps, freq_bins)
        """

        drop_rate = self.drop_rate

        # (samples_num, freq_bins, time_steps)
        x = input.transpose(1, 2)

        # Add an extra dimension for using Conv2d
        # (samples_num, freq_bins, time_steps, 1)
        x = x[:, :, :, None].contiguous()

        a0 = self.bn0(x)
        a1 = F.dropout(F.relu(self.bn1(self.conv1(a0))),
                       p=drop_rate,
                       training=self.training)

        a2 = F.dropout(F.relu(self.bn2(self.conv2(a1))),
                       p=drop_rate,
                       training=self.training)

        emb = F.dropout(F.relu(self.bn3(self.conv3(a2))),
                        p=drop_rate,
                        training=self.training)

        if return_layers is False:
            # (samples_num, hidden_units, time_steps, 1)
            return emb

        else:
            return [a0, a1, a2, emb]


class DecisionLevelMaxPooling(nn.Module):

    def __init__(self, freq_bins, classes_num, hidden_units, drop_rate):

        super(DecisionLevelMaxPooling, self).__init__()

        self.emb = EmbeddingLayers(freq_bins, hidden_units, drop_rate)
        self.fc_final = nn.Linear(hidden_units, classes_num)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, time_steps, freq_bins)
        """

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(input)

        # (samples_num, time_steps, hidden_units)
        b1 = b1[:, :, :, 0].transpose(1, 2)

        b2 = F.sigmoid(self.fc_final(b1))

        # (samples_num, classes_num)
        (output, _) = torch.max(b2, dim=1)

        return output


class DecisionLevelAveragePooling(nn.Module):

    def __init__(self, freq_bins, classes_num, hidden_units, drop_rate):

        super(DecisionLevelAveragePooling, self).__init__()

        self.emb = EmbeddingLayers(freq_bins, hidden_units, drop_rate)
        self.fc_final = nn.Linear(hidden_units, classes_num)

    def init_weights(self):

        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(input)

        # (samples_num, time_steps, hidden_units)
        b1 = b1[:, :, :, 0].transpose(1, 2)

        b2 = F.sigmoid(self.fc_final(b1))

        # (samples_num, classes_num)
        output = torch.mean(b2, dim=1)

        return output


class DecisionLevelSingleAttention(nn.Module):

    def __init__(self, freq_bins, classes_num, hidden_units, drop_rate):

        super(DecisionLevelSingleAttention, self).__init__()

        self.emb = EmbeddingLayers(freq_bins, hidden_units, drop_rate)
        self.attention = Attention(
            hidden_units,
            classes_num,
            att_activation='sigmoid',
            cla_activation='sigmoid')

    def init_weights(self):
        pass

    def forward(self, input):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(input)

        # (samples_num, classes_num, time_steps, 1)
        output = self.attention(b1)

        return output


class DecisionLevelMultiAttention(nn.Module):

    def __init__(self, freq_bins, classes_num, hidden_units, drop_rate):

        super(DecisionLevelMultiAttention, self).__init__()

        self.emb = EmbeddingLayers(freq_bins, hidden_units, drop_rate)
        self.attention = Attention(
            hidden_units,
            classes_num,
            att_activation='sigmoid',
            cla_activation='sigmoid')
        self.fc_final = nn.Linear(classes_num * 2, classes_num)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        # (samples_num, hidden_units, time_steps, 1)
        emb_layers = self.emb(input, return_layers=True)

        # (samples_num, classes_num)
        output1 = self.attention(emb_layers[-1])
        output2 = self.attention(emb_layers[-2])

        # (samples_num, classes_num * 2)
        cat_output = torch.cat((output1, output2), dim=1)

        # (samples_num, class_num)
        output = F.sigmoid(self.fc_final(cat_output))

        return output


class FeatureLevelSingleAttention(nn.Module):

    def __init__(self, freq_bins, classes_num, hidden_units, drop_rate):

        super(FeatureLevelSingleAttention, self).__init__()

        self.emb = EmbeddingLayers(freq_bins, hidden_units, drop_rate)
        
        self.attention = Attention(
            hidden_units,
            hidden_units,
            att_activation='sigmoid',
            cla_activation='linear')
            
        self.fc_final = nn.Linear(hidden_units, classes_num)
        self.bn_attention = nn.BatchNorm1d(hidden_units)

        self.drop_rate = drop_rate

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc_final)
        init_bn(self.bn_attention)

    def forward(self, input):
        """input: (samples_num, freq_bins, time_steps, 1)
        """
        drop_rate = self.drop_rate

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(input)

        # (samples_num, hidden_units)
        b2 = self.attention(b1)
        b2 = F.dropout(
            F.relu(
                self.bn_attention(b2)),
            p=drop_rate,
            training=self.training)

        # (samples_num, classes_num)
        output = F.sigmoid(self.fc_final(b2))

        return output


def train(args):

    model_type = args.model_type

    freq_bins = 128
    classes_num = 527

    # Hyper parameters
    hidden_units = 1024
    drop_rate = 0.5
    batch_size = 500

    if model_type == 'decision_level_max_pooling':
        '''Global max pooling.

        [1] Choi, Keunwoo, et al. "Automatic tagging using deep convolutional
        neural networks." arXiv preprint arXiv:1606.00298 (2016).
        '''
        model = DecisionLevelMaxPooling(
            freq_bins, classes_num, hidden_units, drop_rate)

    elif model_type == 'decision_level_average_pooling':
        '''Global average pooling.

        [2] Lin, Min, et al. Qiang Chen, and Shuicheng Yan. "Network in
        network." arXiv preprint arXiv:1312.4400 (2013).
        '''
        model = DecisionLevelAveragePooling(
            freq_bins, classes_num, hidden_units, drop_rate)

    elif model_type == 'decision_level_single_attention':
        '''Decision level single attention pooling.

        [3] Kong, Qiuqiang, et al. "Audio Set classification with attention
        model: A probabilistic perspective." arXiv preprint arXiv:1711.00927
        (2017).
        '''
        model = DecisionLevelSingleAttention(
            freq_bins, classes_num, hidden_units, drop_rate)

    elif model_type == 'decision_level_multi_attention':
        '''Decision level multi attention pooling.

        [4] Yu, Changsong, et al. "Multi-level Attention Model for Weakly
        Supervised Audio Classification." arXiv preprint arXiv:1803.02353
        (2018).
        '''
        model = DecisionLevelMultiAttention(
            freq_bins, classes_num, hidden_units, drop_rate)

    elif model_type == 'feature_level_single_attention':
        '''Decision level multi attention pooling.

        [5] To be appear.
        '''
        model = FeatureLevelSingleAttention(
            freq_bins, classes_num, hidden_units, drop_rate)

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
                                 'feature_level_single_attention'])

    parser.add_argument('--learning_rate', type=float, default=1e-3)

    subparsers = parser.add_subparsers(dest='mode')
    parser_train = subparsers.add_parser('train')
    parser_get_avg_stats = subparsers.add_parser('get_avg_stats')

    args = parser.parse_args()

    args.filename = utilities.get_filename(__file__)

    # Logs
    sub_dir = os.path.join(args.filename,
                           'balance_type={}'.format(args.balance_type),
                           'model_type={}'.format(args.model_type))

    logs_dir = os.path.join(args.workspace, 'logs', sub_dir)
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
