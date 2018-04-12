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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

try:
    import cPickle
except BaseException:
    import _pickle as cPickle


def move_data_to_gpu(x, cuda, volatile=False):
    x = torch.Tensor(x)
    if cuda:
        x = x.cuda()
    x = Variable(x, volatile=volatile)
    return x


def forward_in_batch(model, x, batch_size, cuda):
    model.eval()
    batch_num = int(np.ceil(len(x) / float(batch_size)))
    output_all = []

    for i1 in range(batch_num):
        batch_x = x[i1 * batch_size: (i1 + 1) * batch_size]
        batch_x = move_data_to_gpu(batch_x, cuda, volatile=True)
        output = model(batch_x)
        output_all.append(output)

    output_all = torch.cat(output_all, dim=0)
    return output_all


def evaluate(model, input, target, stats_dir, probs_dir, iteration):
    """Evaluate a model.

    Args:
      model: object
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)
      stats_dir: str, directory to write out statistics.
      probs_dir: str, directory to write out output (samples_num, classes_num)
      iteration: int

    Returns:
      None
    """
    # Check if cuda
    cuda = next(model.parameters()).is_cuda

    utilities.create_folder(stats_dir)
    utilities.create_folder(probs_dir)

    # Predict presence probabilittarget
    callback_time = time.time()
    (clips_num, time_steps, freq_bins) = input.shape

    (input, target) = utilities.transform_data(input, target)

    output = forward_in_batch(model, input, batch_size=500, cuda=cuda)
    output = output.data.cpu().numpy()  # (clips_num, classes_num)

    # Write out presence probabilities
    prob_path = os.path.join(probs_dir, "prob_{}_iters.p".format(iteration))
    cPickle.dump(output, open(prob_path, 'wb'))

    # Calculate statistics
    stats = utilities.calculate_stats(output, target)

    # Write out statistics
    stat_path = os.path.join(stats_dir, "stat_{}_iters.p".format(iteration))
    cPickle.dump(stats, open(stat_path, 'wb'))

    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    logging.info(
        "mAP: {:.6f}, AUC: {:.6f}, Callback time: {:.3f} s".format(
            mAP, mAUC, time.time() - callback_time))

    if False:
        logging.info("Saveing prob to {}".format(prob_path))
        logging.info("Saveing stat to {}".format(stat_path))


def train(args):
    """Train a model.
    """

    data_dir = args.data_dir
    workspace = args.workspace
    mini_data = args.mini_data
    balance_type = args.balance_type
    learning_rate = args.learning_rate
    filename = args.filename
    model_type = args.model_type
    model = args.model
    batch_size = args.batch_size
    cuda = True

    # Move model to gpu
    if cuda:
        model.cuda()

    # Path of hdf5 data
    bal_train_hdf5_path = os.path.join(data_dir, "bal_train.h5")
    unbal_train_hdf5_path = os.path.join(data_dir, "unbal_train.h5")
    test_hdf5_path = os.path.join(data_dir, "eval.h5")

    # Load data
    load_time = time.time()

    if mini_data:
        # Only load balanced data
        (bal_train_x, bal_train_y, bal_train_id_list) = utilities.load_data(
            bal_train_hdf5_path)

        train_x = bal_train_x
        train_y = bal_train_y
        train_id_list = bal_train_id_list

    else:
        # Load both balanced and unbalanced data
        (bal_train_x, bal_train_y, bal_train_id_list) = utilities.load_data(
            bal_train_hdf5_path)

        (unbal_train_x, unbal_train_y, unbal_train_id_list) = utilities.load_data(
            unbal_train_hdf5_path)

        train_x = np.concatenate((bal_train_x, unbal_train_x))
        train_y = np.concatenate((bal_train_y, unbal_train_y))
        train_id_list = bal_train_id_list + unbal_train_id_list

    # Test data
    (test_x, test_y, test_id_list) = utilities.load_data(test_hdf5_path)

    logging.info("Loading data time: {:.3f} s".format(time.time() - load_time))
    logging.info("Training data shape: {}".format(train_x.shape))

    # Optimization method
    optimizer = optim.Adam(model.parameters(),
                           lr=1e-3,
                           betas=(0.9, 0.999),
                           eps=1e-07)

    # Output directories
    sub_dir = os.path.join(filename,
                           'balance_type={}'.format(balance_type),
                           'model_type={}'.format(model_type))

    models_dir = os.path.join(workspace, "models", sub_dir)
    utilities.create_folder(models_dir)

    stats_dir = os.path.join(workspace, "stats", sub_dir)
    utilities.create_folder(stats_dir)

    probs_dir = os.path.join(workspace, "probs", sub_dir)
    utilities.create_folder(probs_dir)

    # Data generator
    if balance_type == 'no_balance':
        DataGenerator = data_generator.VanillaDataGenerator

    elif balance_type == 'balance_in_batch':
        DataGenerator = data_generator.BalancedDataGenerator

    else:
        raise Exception("Incorrect balance_type!")

    train_gen = DataGenerator(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        shuffle=True,
        seed=1234)

    iteration = 0
    call_freq = 1000
    train_time = time.time()

    for (batch_x, batch_y) in train_gen.generate():

        # Compute stats every several interations
        if iteration % call_freq == 0 and iteration > 1:

            logging.info("------------------")

            logging.info(
                "Iteration: {}, train time: {:.3f} s".format(
                    iteration, time.time() - train_time))

            logging.info("Balance train statistics:")
            evaluate(
                model=model,
                input=bal_train_x,
                target=bal_train_y,
                stats_dir=os.path.join(stats_dir, 'bal_train'),
                probs_dir=os.path.join(probs_dir, 'bal_train'),
                iteration=iteration)

            logging.info("Test statistics:")
            evaluate(
                model=model,
                input=test_x,
                target=test_y,
                stats_dir=os.path.join(stats_dir, "test"),
                probs_dir=os.path.join(probs_dir, "test"),
                iteration=iteration)

            train_time = time.time()

        (batch_x, batch_y) = utilities.transform_data(batch_x, batch_y)

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        # Forward.
        model.train()
        output = model(batch_x)

        # Loss.
        loss = F.binary_cross_entropy(output, batch_y)

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration += 1

        # Save model.
        if iteration % 5000 == 0:
            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(), }
            save_out_path = os.path.join(
                models_dir, "md_{}_iters.tar".format(iteration))
            torch.save(save_out_dict, save_out_path)
            logging.info("Save model to {}".format(save_out_path))

        # Stop training when maximum iteration achieves
        if iteration == 50001:
            break
