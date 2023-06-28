import argparse
import multiprocessing as mp
import os
import pickle
from sklearn.datasets import load_svmlight_file

import numpy as np

from logistic import LogisticDecentralizedSGD
from parameters import Parameters
from utils import pickle_it

from experiment import run_logistic, multi_run_experiment

A, b = None, None

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('experiment', type=str)
    # args = parser.parse_args()
    # print(args)
    # assert args.experiment in ['final']

    dataset_path = os.path.expanduser('G:/Sweden Work/My codes/data/epsilon.pickle')
    n, d = 400000, 2000

    ###############################################
    ### RANDOM DATA PARTITION #####################
    ###############################################
    n_cores = 9
    ################### FINAL ################################

    split_way = 'random'
    split_name = split_way
    random_seed = 1
    num_epoch = 10
    n_repeat = 5
    cores = 100
    experiment = 'final'
    if experiment in ['final']:
        params = []
        for i in np.arange(0, n_repeat+1):
            params += [
                Parameters(name="decentralized-exact", num_epoch=num_epoch,
                           lr_type='decay', initial_lr=0.1, tau=d,
                           regularizer=1 / n, quantization='full',
                           n_cores=n_cores, method='plain',
                           split_data_random_seed=random_seed,
                           distribute_data=True, split_data_strategy=split_name,
                           topology='ring', estimate='final'),
            ]
        multi_run_experiment("dump/aaaepsilon-final-decentralized-" + split_way + "-" + \
                       str(n_cores) + "/", dataset_path, params, nproc=10)

    # if experiment in ['final']:
    #     params = []
    #     for i in np.arange(0, n_repeat + 1):
    #         params += [Parameters(name="new-qsgd", num_epoch=num_epoch,
    #                             lr_type='decay', initial_lr=0.1, tau=d,
    #                             regularizer=1 / n, consensus_lr=0.04,
    #                             quantization='new-qsgd-unbiased', num_levels=32,
    #                             n_cores=cores, method='new', topology='star',
    #                             estimate='final', split_data_random_seed=random_seed,
    #                             distribute_data=True, split_data_strategy=split_name, probability=1)]
    #         multi_run_experiment("dump/epsilon-final-new-qsgd-5-20-epoch-bit-" + split_way + "-" + str(cores) \
    #                        + "-star" + " prob = " + "1" + "/", dataset_path, params, nproc=10)