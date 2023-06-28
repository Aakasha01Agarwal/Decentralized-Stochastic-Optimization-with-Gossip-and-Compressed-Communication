import argparse
import multiprocessing as mp
import os
import pickle
from sklearn.datasets import load_svmlight_file

import numpy as np

from logistic import LogisticDecentralizedSGD
from parameters import Parameters
from utils import pickle_it

from experiment import run_logistic, run_experiment

A, b = None, None

if __name__ == "__main__":
  # parser = argparse.ArgumentParser()
  # parser.add_argument('experiment', type=str)
  # args = parser.parse_args()
  # print(args)
  # assert args.experiment in ['final']

  dataset_path = os.path.expanduser('./data/epsilon.pickle')
  n, d = 400000, 2000

###############################################
### RANDOM DATA PARTITION #####################
###############################################
  n_cores =  [ 100]
################### FINAL ################################

  split_way = 'random'
  split_name = split_way

  num_epoch = 10
  n_repeat = 1
  random_seed = 1
  x = 'final'


  # New method with edge activation

  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #           params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                                lr_type='decay', initial_lr=0.1, tau=d,
  #                                regularizer=1 / n, consensus_lr=0.04,
  #                                quantization='new-qsgd-unbiased', num_levels = 4,
  #                                n_cores=cores, method='new', topology='star',
  #                                estimate='final', split_data_random_seed=random_seed,
  #                                distribute_data=True, split_data_strategy=split_name, probability = 0.5)
  #           run_experiment("dump/epsilon-final-new1-qsgd-2-bit-" + split_way + "-" + str(cores)\
  #                        + "-star"+" prob = "+"0.5"+"/", dataset_path, params, n_repeat, nproc=1)
  #
  # #  p = 0.1, n_bits = 8
  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #           params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                                lr_type='decay', initial_lr=0.1, tau=d,
  #                                regularizer=1 / n, consensus_lr=0.04,
  #                                quantization='new-qsgd-unbiased', num_levels = 16,
  #                                n_cores=cores, method='new', topology='star',
  #                                estimate='final', split_data_random_seed=random_seed,
  #                                distribute_data=True, split_data_strategy=split_name, probability = 0.5)
  #           run_experiment("dump/epsilon-final-new-qsgd-4-bit-" + split_way + "-" + str(cores)\
  #                        + "-star"+" prob = "+"0.5"+"/", dataset_path, params, n_repeat, nproc=1)
  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #           params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                                lr_type='decay', initial_lr=0.1, tau=d,
  #                                regularizer=1 / n, consensus_lr=0.04,
  #                                quantization='new-qsgd-unbiased', num_levels = 32,
  #                                n_cores=cores, method='new', topology='star',
  #                                estimate='final', split_data_random_seed=random_seed,
  #                                distribute_data=True, split_data_strategy=split_name, probability = 0.25)
  #           run_experiment("dump/epsilon-final-new-qsgd-5-bit-20-epoch-" + split_way + "-" + str(cores)\
  #                        + "-star"+" prob = "+"0.25"+"/", dataset_path, params, n_repeat, nproc=1)
  # #  p = 1, n_bits = 8
  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #           params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                                lr_type='decay', initial_lr=0.1, tau=d,
  #                                regularizer=1 / n, consensus_lr=0.04,
  #                                quantization='new-qsgd-unbiased', num_levels = 64,
  #                                n_cores=cores, method='new', topology='star',
  #                                estimate='final', split_data_random_seed=random_seed,
  #                                distribute_data=True, split_data_strategy=split_name, probability = 0.5)
  #           run_experiment("dump/epsilon-final-new1-qsgd-6-bit-" + split_way + "-" + str(cores)\
  #                        + "-star"+" prob = "+"0.5"+"/", dataset_path, params, n_repeat, nproc=1)
  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #           params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                                lr_type='decay', initial_lr=0.1, tau=d,
  #                                regularizer=1 / n, consensus_lr=0.04,
  #                                quantization='new-qsgd-unbiased', num_levels = 128,
  #                                n_cores=cores, method='new', topology='star',
  #                                estimate='final', split_data_random_seed=random_seed,
  #                                distribute_data=True, split_data_strategy=split_name, probability = 0.5)
  #           run_experiment("dump/epsilon-final-new1-qsgd-7-" + split_way + "-" + str(cores)\
  #                        + "-star"+" prob = "+"0.5"+"/", dataset_path, params, n_repeat, nproc=1)

  # #  p = 0.5, n_bits = 8
  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #           params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                                lr_type='decay', initial_lr=0.1, tau=d,
  #                                regularizer=1 / n, consensus_lr=0.04,
  #                                quantization='new-qsgd-unbiased', num_levels = 64,
  #                                n_cores=cores, method='new', topology='star',
  #                                estimate='final', split_data_random_seed=random_seed,
  #                                distribute_data=True, split_data_strategy=split_name, probability = 0.5)
  #           run_experiment("dump/epsilon-final-new1-qsgd-6-bit-" + split_way + "-" + str(cores)\
  #                        + "-star"+" prob = "+"0.5"+"/", dataset_path, params, n_repeat, nproc=1)
  # # #  p = 0.25, n_bits = 8
  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #           params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                                lr_type='decay', initial_lr=0.1, tau=d,
  #                                regularizer=1 / n, consensus_lr=0.04,
  #                                quantization='new-qsgd-unbiased', num_levels = 64,
  #                                n_cores=cores, method='new', topology='star',
  #                                estimate='final', split_data_random_seed=random_seed,
  #                                distribute_data=True, split_data_strategy=split_name, probability = 0.25)
  #           run_experiment("dump/epsilon-final-new1-qsgd-6-bit-" + split_way + "-" + str(cores)\
  #                        + "-star"+" prob = "+"0.25"+"/", dataset_path, params, n_repeat, nproc=1)

  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #         params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                             lr_type='decay', initial_lr=0.1, tau=d,
  #                             regularizer=1 / n, consensus_lr=1,
  #                             quantization='new-qsgd-unbiased', num_levels=32,
  #                             n_cores=cores, method='new', topology='star',
  #                             estimate='final', split_data_random_seed=random_seed,
  #                             distribute_data=True, split_data_strategy=split_name, probability=0.2)
  #         run_experiment("dump/epsilon-final-new-qsgd-5-bit-" + split_way + "-" + str(cores) \
  #                        + "-centralized" + " prob = " + "0.2" + "/", dataset_path, params, n_repeat, nproc=1)
  #
  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #         params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                             lr_type='decay', initial_lr=0.1, tau=d,
  #                             regularizer=1 / n, consensus_lr=1,
  #                             quantization='new-qsgd-unbiased', num_levels=32,
  #                             n_cores=cores, method='new', topology='star',
  #                             estimate='final', split_data_random_seed=random_seed,
  #                             distribute_data=True, split_data_strategy=split_name, probability=0.15)
  #         run_experiment("dump/epsilon-final-new-qsgd-5-bit-" + split_way + "-" + str(cores) \
  #                        + "-centralized" + " prob = " + "0.15" + "/", dataset_path, params, n_repeat, nproc=1)


  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #         params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                             lr_type='decay', initial_lr=0.1, tau=d,
  #                             regularizer=1 / n, consensus_lr=0.04,
  #                             quantization='new-qsgd-unbiased', num_levels=32,
  #                             n_cores=cores, method='new', topology='star',
  #                             estimate='final', split_data_random_seed=random_seed,
  #                             distribute_data=True, split_data_strategy=split_name, probability=0.4)
  #         run_experiment("dump/epsilon-final-new-qsgd-5-bit-" + split_way + "-" + str(cores) \
  #                        + "-star" + " prob = " + "0.4" + "/", dataset_path, params, n_repeat, nproc=1)

  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #         params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                             lr_type='decay', initial_lr=0.1, tau=d,
  #                             regularizer=1 / n, consensus_lr=0.04,
  #                             quantization='new-qsgd-unbiased', num_levels=4,
  #                             n_cores=cores, method='new', topology='star',
  #                             estimate='final', split_data_random_seed=random_seed,
  #                             distribute_data=True, split_data_strategy=split_name, probability=0.5)
  #         run_experiment("dump/epsilon-final-new-qsgd-2-bit-" + split_way + "-" + str(cores) \
  #                        + "-star" + " prob = " + "0.5" + "/", dataset_path, params, n_repeat, nproc=1)
  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #         params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                             lr_type='decay', initial_lr=0.1, tau=d,
  #                             regularizer=1 / n, consensus_lr=0.04,
  #                             quantization='new-qsgd-unbiased', num_levels=16,
  #                             n_cores=cores, method='new', topology='star',
  #                             estimate='final', split_data_random_seed=random_seed,
  #                             distribute_data=True, split_data_strategy=split_name, probability=0.5)
  #         run_experiment("dump/epsilon-final-new-qsgd-4-bit-" + split_way + "-" + str(cores) \
  #                        + "-star" + " prob = " + "0.5" + "/", dataset_path, params, n_repeat, nproc=1)

  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #         params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                             lr_type='decay', initial_lr=0.1, tau=d,
  #                             regularizer=1 / n, consensus_lr=0.04,
  #                             quantization='new-qsgd-unbiased', num_levels=1028,
  #                             n_cores=cores, method='new', topology='star',
  #                             estimate='final', split_data_random_seed=random_seed,
  #                             distribute_data=True, split_data_strategy=split_name, probability=0.4)
  #         run_experiment("dump/epsilon-final-new-qsgd-10-bit-" + split_way + "-" + str(cores) \
  #                        + "-star" + " prob = " + "0.4" + "/", dataset_path, params, n_repeat, nproc=1)
  #
  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #         params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                             lr_type='decay', initial_lr=0.1, tau=d,
  #                             regularizer=1 / n, consensus_lr=0.04,
  #                             quantization='new-qsgd-unbiased', num_levels=32,
  #                             n_cores=cores, method='new', topology='star',
  #                             estimate='final', split_data_random_seed=random_seed,
  #                             distribute_data=True, split_data_strategy=split_name, probability=0.8)
  #         run_experiment("dump/epsilon-final-new-qsgd-5-bit-" + split_way + "-" + str(cores) \
  #                        + "-star" + " prob = " + "0.8" + "/", dataset_path, params, n_repeat, nproc=1)


  #     # #  p = 1, n_bits = 8
  if x in ['final']:
      params = []
      for cores in n_cores:
          params = Parameters(name="new-qsgd", num_epoch=num_epoch,
                              lr_type='decay', initial_lr=0.1, tau=d,
                              regularizer=1 / n, consensus_lr=0.04,
                              quantization='new-qsgd-unbiased', num_levels=32,
                              n_cores=cores, method='new', topology='centralized',
                              estimate='final', split_data_random_seed=random_seed,
                              distribute_data=True, split_data_strategy=split_name, probability=1)
          run_experiment("dump/epsilon-final-new-qsgd-5-bit-" + split_way + "-" + str(cores) \
                         + "-centralized" + " prob = " + "1" + "/", dataset_path, params, n_repeat, nproc=1)
  #     #  p = 0.5, n_bits = 8
  if x in ['final']:
      params = []
      for cores in n_cores:
          params = Parameters(name="new-qsgd", num_epoch=num_epoch,
                              lr_type='decay', initial_lr=0.1, tau=d,
                              regularizer=1 / n, consensus_lr=0.04,
                              quantization='new-qsgd-unbiased', num_levels=32,
                              n_cores=cores, method='new', topology='centralized',
                              estimate='final', split_data_random_seed=random_seed,
                              distribute_data=True, split_data_strategy=split_name, probability=0.5)
          run_experiment("dump/epsilon-final-new-qsgd-5-bit-" + split_way + "-" + str(cores) \
                         + "-centralized" + " prob = " + "0.5" + "/", dataset_path, params, n_repeat, nproc=1)
  #     # #  p = 0.25, n_bits = 8
  if x in ['final']:
      params = []
      for cores in n_cores:
          params = Parameters(name="new-qsgd", num_epoch=num_epoch,
                              lr_type='decay', initial_lr=0.1, tau=d,
                              regularizer=1 / n, consensus_lr=0.04,
                              quantization='new-qsgd-unbiased', num_levels=32,
                              n_cores=cores, method='new', topology='centralized',
                              estimate='final', split_data_random_seed=random_seed,
                              distribute_data=True, split_data_strategy=split_name, probability=0.25)
          run_experiment("dump/epsilon-final-new-qsgd-5-bit-" + split_way + "-" + str(cores) \
                         + "-centralized" + " prob = " + "0.25" + "/", dataset_path, params, n_repeat, nproc=1)


  if x in ['final']:
      params = []
      for cores in n_cores:
          params = Parameters(name="new-qsgd", num_epoch=num_epoch,
                              lr_type='decay', initial_lr=0.1, tau=d,
                              regularizer=1 / n, consensus_lr=0.04,
                              quantization='new-qsgd-unbiased', num_levels=32,
                              n_cores=cores, method='new', topology='centralized',
                              estimate='final', split_data_random_seed=random_seed,
                              distribute_data=True, split_data_strategy=split_name, probability=0)
          run_experiment("dump/epsilon-final-new-qsgd-5-bit-" + split_way + "-" + str(cores) \
                         + "-centralized" + " prob = " + "0" + "/", dataset_path, params, n_repeat, nproc=1)

  #  p = 0.5 n_bits = 4

  # p = 0.25 n_bits = 2



  # NEW METHOD
  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #           params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                                lr_type='decay', initial_lr=0.1, tau=d,
  #                                regularizer=1 / n, consensus_lr=0.04,
  #                                quantization='new-qsgd-unbiased', num_levels = 256,
  #                                n_cores=cores, method='new', topology='ring',
  #                                estimate='final', split_data_random_seed=random_seed,
  #                                distribute_data=True, split_data_strategy=split_name)
  #           run_experiment("dump/epsilon-final-new-qsgd-8-bit" + split_way + "-" + str(cores)\
  #                        + "-ring"+"/", dataset_path, params, n_repeat, nproc=1)
  #
  #
  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #           params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                                lr_type='decay', initial_lr=0.1, tau=d,
  #                                regularizer=1 / n, consensus_lr=0.04,
  #                                quantization='new-qsgd-unbiased', num_levels = 256,
  #                                n_cores=cores, method='new', topology='star',
  #                                estimate='final', split_data_random_seed=random_seed,
  #                                distribute_data=True, split_data_strategy=split_name)
  #           run_experiment("dump/epsilon-final-new-qsgd-8-bit" + split_way + "-" + str(cores)\
  #                        + "-star"+"/", dataset_path, params, n_repeat, nproc=1)
  #
  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #           params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                                lr_type='decay', initial_lr=0.1, tau=d,
  #                                regularizer=1 / n, consensus_lr=0.04,
  #                                quantization='new-qsgd-unbiased', num_levels = 256,
  #                                n_cores=cores, method='new', topology='random',
  #                                estimate='final', split_data_random_seed=random_seed,
  #                                distribute_data=True, split_data_strategy=split_name)
  #           run_experiment("dump/epsilon-final-new-qsgd-8-bit" + split_way + "-" + str(cores)\
  #                        + "-random"+"/", dataset_path, params, n_repeat, nproc=1)


  # 4 BITS
  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #         params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                             lr_type='decay', initial_lr=0.1, tau=d,
  #                             regularizer=1 / n, consensus_lr=0.04,
  #                             quantization='new-qsgd-unbiased', num_levels=16,
  #                             n_cores=cores, method='new', topology='centralized',
  #                             estimate='final', split_data_random_seed=random_seed,
  #                             distribute_data=True, split_data_strategy=split_name)
  #         run_experiment("dump/epsilon-final-new-qsgd-4-bit" + split_way + "-" + str(cores) \
  #                        + "-ring" + "/", dataset_path, params, n_repeat, nproc=1)

  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #         params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                             lr_type='decay', initial_lr=0.1, tau=d,
  #                             regularizer=1 / n, consensus_lr=0.04,
  #                             quantization='new-qsgd-unbiased', num_levels=16,
  #                             n_cores=cores, method='new', topology='centralized',
  #                             estimate='final', split_data_random_seed=random_seed,
  #                             distribute_data=True, split_data_strategy=split_name)
  #         run_experiment("dump/epsilon-final-new-qsgd-4-bit" + split_way + "-" + str(cores) \
  #                        + "-star" + "/", dataset_path, params, n_repeat, nproc=1)
  #
  # if x in ['final']:
  #     params = []
  #     for cores in n_cores:
  #         params = Parameters(name="new-qsgd", num_epoch=num_epoch,
  #                             lr_type='decay', initial_lr=0.1, tau=d,
  #                             regularizer=1 / n, consensus_lr=0.04,
  #                             quantization='new-qsgd-unbiased', num_levels=16,
  #                             n_cores=cores, method='new', topology='centralized',
  #                             estimate='final', split_data_random_seed=random_seed,
  #                             distribute_data=True, split_data_strategy=split_name)
  #         run_experiment("dump/epsilon-final-new-qsgd-4-bit" + split_way + "-" + str(cores) \
  #                        + "-random" + "/", dataset_path, params, n_repeat, nproc=1)



  # if x in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #       for n_core in n_cores:
  #         params = Parameters(name="decentralized-exact", num_epoch=num_epoch,
  #                              lr_type='decay', initial_lr=0.1, tau=d,
  #                              regularizer=1 / n, quantization='full',
  #                              n_cores=n_core, method='plain',
  #                              split_data_random_seed=random_seed,
  #                              distribute_data=True, split_data_strategy=split_name,
  #                              topology='ring', estimate='final')
  #
  #         run_experiment("dump/epsilon-final-decentralized-" + split_way + "-" + \
  #                          str(n_core) +  'random' +"-ring"+"/", dataset_path, params,n_repeat, nproc=1)
  # if x in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #       for n_core in n_cores:
  #         params = Parameters(name="decentralized-exact", num_epoch=num_epoch,
  #                              lr_type='decay', initial_lr=0.1, tau=d,
  #                              regularizer=1 / n, quantization='full',
  #                              n_cores=n_core, method='plain',
  #                              split_data_random_seed=random_seed,
  #                              distribute_data=True, split_data_strategy=split_name,
  #                              topology='star', estimate='final')
  #
  #         run_experiment("dump/epsilon-final-decentralized-" + split_way + "-" + \
  #                          str(n_core) +  'random'  +"-star"+ "/", dataset_path, params,n_repeat, nproc=1)
  #
  # if x in ['final']:
  #   params = []
  #   for random_seed in np.arange(1, n_repeat + 1):
  #       for n_core in n_cores:
  #         params = Parameters(name="decentralized-exact", num_epoch=num_epoch,
  #                              lr_type='decay', initial_lr=0.1, tau=d,
  #                              regularizer=1 / n, quantization='full',
  #                              n_cores=n_core, method='plain',
  #                              split_data_random_seed=random_seed,
  #                              distribute_data=True, split_data_strategy=split_name,
  #                              topology='random', estimate='final')
  #
  #         run_experiment("dump/epsilon-final-decentralized-" + split_way + "-" + \
  #                          str(n_core) +  'random'  +"-random"+ "/", dataset_path, params,n_repeat, nproc=1)

  # if x in ['final']:
  #     params = []
  #     for n_core in n_cores:
  #       params = Parameters(name="dcd-psgd-random-20",
  #                            num_epoch=num_epoch, lr_type='decay',
  #                            initial_lr=1e-15, tau=d, regularizer=1 / n,
  #                            quantization='random-unbiased', coordinates_to_keep=20,
  #                            n_cores=n_core, method='dcd-psgd',
  #                            split_data_random_seed=random_seed,
  #                            distribute_data=True, split_data_strategy=split_name,
  #                            topology='ring', estimate='final')
  #       run_experiment("dump/epsilon-final-dcd-random-20-" + split_way + "-" + str(n_core)\
  #                    +"-ring"+ "/", dataset_path, params, n_repeat, nproc=1)
  #
  # if x in ['final']:
  #     params = []
  #     for n_core in n_cores:
  #       params = Parameters(name="dcd-psgd-random-20",
  #                            num_epoch=num_epoch, lr_type='decay',
  #                            initial_lr=1e-15, tau=d, regularizer=1 / n,
  #                            quantization='random-unbiased', coordinates_to_keep=20,
  #                            n_cores=n_core, method='dcd-psgd',
  #                            split_data_random_seed=random_seed,
  #                            distribute_data=True, split_data_strategy=split_name,
  #                            topology='star', estimate='final')
  #       run_experiment("dump/epsilon-final-dcd-random-20-" + split_way + "-" + str(n_core)\
  #                    +"-star"+ "/", dataset_path, params, n_repeat, nproc=1)
  # if x in ['final']:
  #     params = []
  #     for n_core in n_cores:
  #       params = Parameters(name="dcd-psgd-random-20",
  #                            num_epoch=num_epoch, lr_type='decay',
  #                            initial_lr=1e-15, tau=d, regularizer=1 / n,
  #                            quantization='random-unbiased', coordinates_to_keep=20,
  #                            n_cores=n_core, method='dcd-psgd',
  #                            split_data_random_seed=random_seed,
  #                            distribute_data=True, split_data_strategy=split_name,
  #                            topology='random', estimate='final')
  #       run_experiment("dump/epsilon-final-dcd-random-20-" + split_way + "-" + str(n_core)\
  #                    +"-random"+ "/", dataset_path, params, n_repeat, nproc=1)
  # if x in ['final']:
  #     params = []
  #     for n_core in n_cores:
  #       params = Parameters(name="dcd-psgd-top-20",
  #                            num_epoch=num_epoch, lr_type='decay',
  #                            initial_lr=1e-15, tau=d, regularizer=1 / n,
  #                            quantization='top', coordinates_to_keep=20,
  #                            n_cores=n_core, method='dcd-psgd',
  #                            split_data_random_seed=random_seed,
  #                            distribute_data=True, split_data_strategy=split_name,
  #                            topology='ring', estimate='final')
  #       run_experiment("dump/epsilon-final-dcd-top-20-" + split_way + "-" + str(n_core)\
  #                    +"-ring"+ "/", dataset_path, params, n_repeat, nproc=1)
  # if x in ['final']:
  #     params = []
  #     for n_core in n_cores:
  #       params = Parameters(name="dcd-psgd-top-20",
  #                            num_epoch=num_epoch, lr_type='decay',
  #                            initial_lr=1e-15, tau=d, regularizer=1 / n,
  #                            quantization='top', coordinates_to_keep=20,
  #                            n_cores=n_core, method='dcd-psgd',
  #                            split_data_random_seed=random_seed,
  #                            distribute_data=True, split_data_strategy=split_name,
  #                            topology='star', estimate='final')
  #       run_experiment("dump/epsilon-final-dcd-top-20-" + split_way + "-" + str(n_core)\
  #                    +"-star"+ "/", dataset_path, params, n_repeat, nproc=1)
  # if x in ['final']:
  #     params = []
  #     for n_core in n_cores:
  #       params = Parameters(name="dcd-psgd-top-20",
  #                            num_epoch=num_epoch, lr_type='decay',
  #                            initial_lr=1e-15, tau=d, regularizer=1 / n,
  #                            quantization='top', coordinates_to_keep=20,
  #                            n_cores=n_core, method='dcd-psgd',
  #                            split_data_random_seed=random_seed,
  #                            distribute_data=True, split_data_strategy=split_name,
  #                            topology='random', estimate='final')
  #       run_experiment("dump/epsilon-final-dcd-top-20-" + split_way + "-" + str(n_core)\
  #                    +"-random"+ "/", dataset_path, params, n_repeat, nproc=1)



#   if x in ['final']:
#     params = []
#     # for random_seed in np.arange(1, n_repeat + 1):
#     for n_core in n_cores:
#       params = Parameters(name="dcd-psgd-qsgd", num_epoch=num_epoch,
#                             lr_type='decay', initial_lr=0.01, tau=d,
#                             regularizer=1 / n, quantization='qsgd-unbiased',
#                             num_levels=16, n_cores=n_core, method='dcd-psgd',
#                             split_data_random_seed=random_seed,
#                             distribute_data=True, split_data_strategy=split_name,
#                             topology='ring', estimate='final')
#       run_experiment("dump/epsilon-final-dcd-qsgd-4bit-" + split_way + "-" + str(n_core) \
#                      + '-ring' + "/", dataset_path, params, n_repeat,nproc=1)
#   if x in ['final']:
#       params = []
#       # for random_seed in np.arange(1, n_repeat + 1):
#       for n_core in n_cores:
#           params = Parameters(name="dcd-psgd-qsgd", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=0.01, tau=d,
#                               regularizer=1 / n, quantization='qsgd-unbiased',
#                               num_levels=256, n_cores=n_core, method='dcd-psgd',
#                               split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name,
#                               topology='ring', estimate='final')
#           run_experiment("dump/epsilon-final-dcd-qsgd-8bit-" + split_way + "-" + str(n_core) \
#                          + '-ring'  + "/", dataset_path, params, n_repeat, nproc=1)
#   if x in ['final']:
#       params = []
#       # for random_seed in np.arange(1, n_repeat + 1):
#       for n_core in n_cores:
#           params = Parameters(name="dcd-psgd-qsgd", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=0.01, tau=d,
#                               regularizer=1 / n, quantization='qsgd-unbiased',
#                               num_levels=4, n_cores=n_core, method='dcd-psgd',
#                               split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name,
#                               topology='ring', estimate='final')
#           run_experiment("dump/epsilon-final-dcd-qsgd-2bit-" + split_way + "-" + str(n_core) \
#                          + '-ring' + "/", dataset_path, params, n_repeat, nproc=1)
#
#
#   if x in ['final']:
#     params = []
#     # for random_seed in np.arange(1, n_repeat + 1):
#     for n_core in n_cores:
#       params = Parameters(name="dcd-psgd-qsgd", num_epoch=num_epoch,
#                             lr_type='decay', initial_lr=0.01, tau=d,
#                             regularizer=1 / n, quantization='qsgd-unbiased',
#                             num_levels=16, n_cores=n_core, method='dcd-psgd',
#                             split_data_random_seed=random_seed,
#                             distribute_data=True, split_data_strategy=split_name,
#                             topology='star', estimate='final')
#       run_experiment("dump/epsilon-final-dcd-qsgd-4bit-" + split_way + "-" + str(n_core) \
#                      + '-star' + "/", dataset_path, params, n_repeat,nproc=1)
#   if x in ['final']:
#       params = []
#       # for random_seed in np.arange(1, n_repeat + 1):
#       for n_core in n_cores:
#           params = Parameters(name="dcd-psgd-qsgd", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=0.01, tau=d,
#                               regularizer=1 / n, quantization='qsgd-unbiased',
#                               num_levels=256, n_cores=n_core, method='dcd-psgd',
#                               split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name,
#                               topology='star', estimate='final')
#           run_experiment("dump/epsilon-final-dcd-qsgd-8bit-" + split_way + "-" + str(n_core) \
#                          + '-star'+ "/", dataset_path, params, n_repeat, nproc=1)
#   if x in ['final']:
#       params = []
#       # for random_seed in np.arange(1, n_repeat + 1):
#       for n_core in n_cores:
#           params = Parameters(name="dcd-psgd-qsgd", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=0.01, tau=d,
#                               regularizer=1 / n, quantization='qsgd-unbiased',
#                               num_levels=4, n_cores=n_core, method='dcd-psgd',
#                               split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name,
#                               topology='star', estimate='final')
#           run_experiment("dump/epsilon-final-dcd-qsgd-2bit-" + split_way + "-" + str(n_core) \
#                          + '-star' + "/", dataset_path, params, n_repeat, nproc=1)
#
# #
# #
# # RANDOM
#
#   if x in ['final']:
#     params = []
#     # for random_seed in np.arange(1, n_repeat + 1):
#     for n_core in n_cores:
#       params = Parameters(name="dcd-psgd-qsgd", num_epoch=num_epoch,
#                             lr_type='decay', initial_lr=1e-06, tau=d,
#                             regularizer=1 / n, quantization='qsgd-unbiased',
#                             num_levels=16, n_cores=n_core, method='dcd-psgd',
#                             split_data_random_seed=random_seed,
#                             distribute_data=True, split_data_strategy=split_name,
#                             topology='random', estimate='final')
#       run_experiment("dump/epsilon-final-dcd-qsgd-4bit-" + split_way + "-" + str(n_core)\
#                    + '-random'+ "/", dataset_path, params, n_repeat,nproc=1)
#   if x in ['final']:
#       params = []
#       # for random_seed in np.arange(1, n_repeat + 1):
#       for n_core in n_cores:
#           params = Parameters(name="dcd-psgd-qsgd", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=1e-06, tau=d,
#                               regularizer=1 / n, quantization='qsgd-unbiased',
#                               num_levels=256, n_cores=n_core, method='dcd-psgd',
#                               split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name,
#                               topology='random', estimate='final')
#           run_experiment("dump/epsilon-final-dcd-qsgd-8bit-" + split_way + "-" + str(n_core) \
#                          + '-random' + "/", dataset_path, params, n_repeat, nproc=1)
#
#
#   if x in ['final']:
#       params = []
#       # for random_seed in np.arange(1, n_repeat + 1):
#       for n_core in n_cores:
#           params = Parameters(name="dcd-psgd-qsgd", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=1e-06, tau=d,
#                               regularizer=1 / n, quantization='qsgd-unbiased',
#                               num_levels=4, n_cores=n_core, method='dcd-psgd',
#                               split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name,
#                               topology='random', estimate='final')
#           run_experiment("dump/epsilon-final-dcd-qsgd-2bit-" + split_way + "-" + str(n_core) \
#                          + '-random' + "/", dataset_path, params, n_repeat, nproc=1)






  #  ECD


  #
  # if x in ['final']:
  #   params = []
  #   for n_core in n_cores:
  #     params = Parameters(name="ecd-psgd-random", num_epoch=num_epoch,
  #                           lr_type='decay', initial_lr=1e-6, tau=d,
  #                           regularizer=1 / n, consensus_lr=None,
  #                           quantization='random-unbiased',
  #                           coordinates_to_keep=20, n_cores=n_core,
  #                           method='ecd-psgd', split_data_random_seed=random_seed,
  #                           distribute_data=True, split_data_strategy=split_name,
  #                           topology='ring', estimate='final')
  #     run_experiment("dump/epsilon-final-ecd-random-20-" + split_way + "-" + str(n_core)\
  #                  +"-ring"+ "/", dataset_path, params,n_repeat, nproc=1)
  #
  # if x in ['final']:
  #   params = []
  #   for n_core in n_cores:
  #     params = Parameters(name="ecd-psgd-random", num_epoch=num_epoch,
  #                           lr_type='decay', initial_lr=1e-6, tau=d,
  #                           regularizer=1 / n, consensus_lr=None,
  #                           quantization='random-unbiased',
  #                           coordinates_to_keep=20, n_cores=n_core,
  #                           method='ecd-psgd', split_data_random_seed=random_seed,
  #                           distribute_data=True, split_data_strategy=split_name,
  #                           topology='star', estimate='final')
  #     run_experiment("dump/epsilon-final-ecd-random-20-" + split_way + "-" + str(n_core)\
  #                  +"-star"+ "/", dataset_path, params,n_repeat, nproc=1)
  #
  # if x in ['final']:
  #   params = []
  #   for n_core in n_cores:
  #     params = Parameters(name="ecd-psgd-random", num_epoch=num_epoch,
  #                           lr_type='decay', initial_lr=1e-6, tau=d,
  #                           regularizer=1 / n, consensus_lr=None,
  #                           quantization='random-unbiased',
  #                           coordinates_to_keep=20, n_cores=n_core,
  #                           method='ecd-psgd', split_data_random_seed=random_seed,
  #                           distribute_data=True, split_data_strategy=split_name,
  #                           topology='random', estimate='final')
  #     run_experiment("dump/epsilon-final-ecd-random-20-" + split_way + "-" + str(n_core)\
  #                  +"-random"+ "/", dataset_path, params,n_repeat, nproc=1)
  #
  #
  # if x in ['final']:
  #   params = []
  #   for n_core in n_cores:
  #       params =Parameters(name="ecd-psgd-qsgd", num_epoch=num_epoch,
  #                           lr_type='decay', initial_lr=1e-06, tau=d,
  #                           regularizer=1 / n, quantization='qsgd-unbiased',
  #                           num_levels=16, n_cores=n_core, method='ecd-psgd',
  #                           split_data_random_seed=random_seed,
  #                           distribute_data=True, split_data_strategy=split_name,
  #                           topology='ring', estimate='final')
  #       run_experiment("dump/epsilon-final-ecd-qsgd-4bit-" + split_way + "-" + str(n_core) +"-ring"\
  #                  + "/", dataset_path, params,n_repeat, nproc=1)
  #
  # if x in ['final']:
  #   params = []
  #   for n_core in n_cores:
  #       params =Parameters(name="ecd-psgd-qsgd", num_epoch=num_epoch,
  #                           lr_type='decay', initial_lr=1e-06, tau=d,
  #                           regularizer=1 / n, quantization='qsgd-unbiased',
  #                           num_levels=4, n_cores=n_core, method='ecd-psgd',
  #                           split_data_random_seed=random_seed,
  #                           distribute_data=True, split_data_strategy=split_name,
  #                           topology='ring', estimate='final')
  #       run_experiment("dump/epsilon-final-ecd-qsgd-2bit-" + split_way + "-" + str(n_core) +"-ring"\
  #                  + "/", dataset_path, params,n_repeat, nproc=1)
  #
  # if x in ['final']:
  #   params = []
  #   for n_core in n_cores:
  #       params =Parameters(name="ecd-psgd-qsgd", num_epoch=num_epoch,
  #                           lr_type='decay', initial_lr=1e-06, tau=d,
  #                           regularizer=1 / n, quantization='qsgd-unbiased',
  #                           num_levels=256, n_cores=n_core, method='ecd-psgd',
  #                           split_data_random_seed=random_seed,
  #                           distribute_data=True, split_data_strategy=split_name,
  #                           topology='ring', estimate='final')
  #       run_experiment("dump/epsilon-final-ecd-qsgd-8bit-" + split_way + "-" + str(n_core) +"-ring"\
  #                  + "/", dataset_path, params,n_repeat, nproc=1)
  #
  # if x in ['final']:
  #     params = []
  #     for n_core in n_cores:
  #         params = Parameters(name="ecd-psgd-qsgd", num_epoch=num_epoch,
  #                             lr_type='decay', initial_lr=1e-06, tau=d,
  #                             regularizer=1 / n, quantization='qsgd-unbiased',
  #                             num_levels=16, n_cores=n_core, method='ecd-psgd',
  #                             split_data_random_seed=random_seed,
  #                             distribute_data=True, split_data_strategy=split_name,
  #                             topology='star', estimate='final')
  #         run_experiment("dump/epsilon-final-ecd-qsgd-4bit-" + split_way + "-" + str(n_core) + "-star" \
  #                        + "/", dataset_path, params,n_repeat, nproc=1)
  #
  # if x in ['final']:
  #     params = []
  #     for n_core in n_cores:
  #         params = Parameters(name="ecd-psgd-qsgd", num_epoch=num_epoch,
  #                             lr_type='decay', initial_lr=1e-06, tau=d,
  #                             regularizer=1 / n, quantization='qsgd-unbiased',
  #                             num_levels=4, n_cores=n_core, method='ecd-psgd',
  #                             split_data_random_seed=random_seed,
  #                             distribute_data=True, split_data_strategy=split_name,
  #                             topology='star', estimate='final')
  #         run_experiment("dump/epsilon-final-ecd-qsgd-2bit-" + split_way + "-" + str(n_core) + "-star" \
  #                        + "/", dataset_path, params,n_repeat, nproc=1)
  #
  # if x in ['final']:
  #     params = []
  #     for n_core in n_cores:
  #         params = Parameters(name="ecd-psgd-qsgd", num_epoch=num_epoch,
  #                             lr_type='decay', initial_lr=1e-06, tau=d,
  #                             regularizer=1 / n, quantization='qsgd-unbiased',
  #                             num_levels=256, n_cores=n_core, method='ecd-psgd',
  #                             split_data_random_seed=random_seed,
  #                             distribute_data=True, split_data_strategy=split_name,
  #                             topology='star', estimate='final')
  #         run_experiment("dump/epsilon-final-ecd-qsgd-8bit-" + split_way + "-" + str(n_core) + "-star" \
  #                        + "/", dataset_path, params, n_repeat,nproc=1)
  #
  # if x in ['final']:
  #     params = []
  #     for n_core in n_cores:
  #         params = Parameters(name="ecd-psgd-qsgd", num_epoch=num_epoch,
  #                             lr_type='decay', initial_lr=1e-06, tau=d,
  #                             regularizer=1 / n, quantization='qsgd-unbiased',
  #                             num_levels=16, n_cores=n_core, method='ecd-psgd',
  #                             split_data_random_seed=random_seed,
  #                             distribute_data=True, split_data_strategy=split_name,
  #                             topology='random', estimate='final')
  #         run_experiment("dump/epsilon-final-ecd-qsgd-4bit-" + split_way + "-" + str(n_core) + "-random" \
  #                        + "/", dataset_path, params, n_repeat,nproc=1)
  #
  # if x in ['final']:
  #     params = []
  #     for n_core in n_cores:
  #         params = Parameters(name="ecd-psgd-qsgd", num_epoch=num_epoch,
  #                             lr_type='decay', initial_lr=1e-06, tau=d,
  #                             regularizer=1 / n, quantization='qsgd-unbiased',
  #                             num_levels=4, n_cores=n_core, method='ecd-psgd',
  #                             split_data_random_seed=random_seed,
  #                             distribute_data=True, split_data_strategy=split_name,
  #                             topology='random', estimate='final')
  #         run_experiment("dump/epsilon-final-ecd-qsgd-2bit-" + split_way + "-" + str(n_core) + "-random" \
  #                        + "/", dataset_path, params,n_repeat, nproc=1)
  #
  # if x in ['final']:
  #     params = []
  #     for n_core in n_cores:
  #         params = Parameters(name="ecd-psgd-qsgd", num_epoch=num_epoch,
  #                             lr_type='decay', initial_lr=1e-06, tau=d,
  #                             regularizer=1 / n, quantization='qsgd-unbiased',
  #                             num_levels=256, n_cores=n_core, method='ecd-psgd',
  #                             split_data_random_seed=random_seed,
  #                             distribute_data=True, split_data_strategy=split_name,
  #                             topology='random', estimate='final')
  #         run_experiment("dump/epsilon-final-ecd-qsgd-8bit-" + split_way + "-" + str(n_core) + "-random" \
  #                        + "/", dataset_path, params, n_repeat,nproc=1)
  # #
  # #
  #
  #
  # if x in ['final']:
  #   params = []
  #   for n_core in n_cores:
  #     params = Parameters(name="ecd-psgd-top", num_epoch=num_epoch,
  #                           lr_type='decay', initial_lr=1e-6, tau=d,
  #                           regularizer=1 / n, consensus_lr=None,
  #                           quantization='top',
  #                           coordinates_to_keep=20, n_cores=n_core,
  #                           method='ecd-psgd', split_data_random_seed=random_seed,
  #                           distribute_data=True, split_data_strategy=split_name,
  #                           topology='ring', estimate='final')
  #     run_experiment("dump/epsilon-final-ecd-top-20-" + split_way + "-" + str(n_core)\
  #                  +"-ring"+ "/", dataset_path, params,n_repeat, nproc=1)


#   if x in ['final']:
#     params = []
#     for n_core in n_cores:
#       params = Parameters(name="ecd-psgd-top", num_epoch=num_epoch,
#                             lr_type='decay', initial_lr=1e-6, tau=d,
#                             regularizer=1 / n, consensus_lr=None,
#                             quantization='top',
#                             coordinates_to_keep=20, n_cores=n_core,
#                             method='ecd-psgd', split_data_random_seed=random_seed,
#                             distribute_data=True, split_data_strategy=split_name,
#                             topology='star', estimate='final')
#       run_experiment("dump/epsilon-final-ecd-top-20-" + split_way + "-" + str(n_core)\
#                    +"-star"+ "/", dataset_path, params,n_repeat, nproc=1)
#
#
#   if x in ['final']:
#     params = []
#     for n_core in n_cores:
#       params = Parameters(name="ecd-psgd-top", num_epoch=num_epoch,
#                             lr_type='decay', initial_lr=1e-6, tau=d,
#                             regularizer=1 / n, consensus_lr=None,
#                             quantization='top',
#                             coordinates_to_keep=20, n_cores=n_core,
#                             method='ecd-psgd', split_data_random_seed=random_seed,
#                             distribute_data=True, split_data_strategy=split_name,
#                             topology='random', estimate='final')
#       run_experiment("dump/epsilon-final-ecd-top-20-" + split_way + "-" + str(n_core)\
#                    +"-random"+ "/", dataset_path, params,n_repeat, nproc=1)
#
#
#
#
#
# # CHOCO
# #
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params = Parameters(name="decentralized-top-20", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.04,
#                            quantization='top', coordinates_to_keep=20,
#                            n_cores=cores, method='choco', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-20-" + split_way + "-" + str(cores)\
#                    + "-ring"+"/", dataset_path, params, n_repeat, nproc=1)
#
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params = Parameters(name="decentralized-top-20", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.04,
#                            quantization='top', coordinates_to_keep=20,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-20-" + split_way + "-" + str(cores)\
#                    + "-star"+"/", dataset_path, params, n_repeat, nproc=1)
#
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params = Parameters(name="decentralized-top-20", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.04,
#                            quantization='top', coordinates_to_keep=20,
#                            n_cores=cores, method='choco', topology='random',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-20-" + split_way + "-" + str(cores)\
#                    + "-random"+"/", dataset_path, params, n_repeat, nproc=1)
#
#   if x in ['final']:
#       params = []
#       for cores in n_cores:
#           params = Parameters(name="decentralized-top-20", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=0.1, tau=d,
#                               regularizer=1 / n, consensus_lr=0.04,
#                               quantization='random-unbiased', coordinates_to_keep=20,
#                               n_cores=cores, method='choco', topology='ring',
#                               estimate='final', split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name)
#           run_experiment("dump/epsilon-final-choco-random-20-" + split_way + "-" + str(cores) \
#                          + "-ring" + "/", dataset_path, params, n_repeat, nproc=1)
# #
#   if x in ['final']:
#       params = []
#       for cores in n_cores:
#           params = Parameters(name="decentralized-top-20", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=0.1, tau=d,
#                               regularizer=1 / n, consensus_lr=0.04,
#                               quantization='random-unbiased', coordinates_to_keep=20,
#                               n_cores=cores, method='choco', topology='star',
#                               estimate='final', split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name)
#           run_experiment("dump/epsilon-final-choco-random-20-" + split_way + "-" + str(cores) \
#                          + "-star" + "/", dataset_path, params, n_repeat, nproc=1)
# #
#   if x in ['final']:
#       params = []
#       for cores in n_cores:
#           params = Parameters(name="decentralized-top-20", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=0.1, tau=d,
#                               regularizer=1 / n, consensus_lr=0.04,
#                               quantization='random-unbiased', coordinates_to_keep=20,
#                               n_cores=cores, method='choco', topology='random',
#                               estimate='final', split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name)
#           run_experiment("dump/epsilon-final-choco-random-20-" + split_way + "-" + str(cores) \
#                          + "-random" + "/", dataset_path, params, n_repeat, nproc=1)
#
#
#
# #
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=20,
#                            quantization='qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-random-4bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=20,
#                            quantization='qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='choco', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-random-4bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=20,
#                            quantization='qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-random-4bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   #
#   #
#   #
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=20,
#                            quantization='qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-random-2bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=20,
#                            quantization='qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='choco', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-random-2bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=20,
#                            quantization='qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-random-2bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#
#   #
#   #
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=20,
#                            quantization='qsgd-unbiased', num_levels=256,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-random-8bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=20,
#                            quantization='qsgd-unbiased', num_levels=256,
#                            n_cores=cores, method='choco', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-random-8bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,
#                            quantization='qsgd-unbiased', num_levels=256,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-random-8bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)



  # if x in ['final']:
  #   params = []
  #   for cores in n_cores:
  #     params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=1000,
  #                          quantization='top-qsgd-unbiased', num_levels=256,
  #                          n_cores=cores, method='choco', topology='star',
  #                          estimate='final', split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name)
  #     run_experiment("dump/epsilon-final-choco-top-8bit-" + split_way + "-" + str(cores)+'-star'\
  #                  + "/", dataset_path, params,n_repeat, nproc=1)
  # if x in ['final']:
  #   params = []
  #   for cores in n_cores:
  #     params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=1000,
  #                          quantization='top-qsgd-unbiased', num_levels=256,
  #                          n_cores=cores, method='choco', topology='ring',
  #                          estimate='final', split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name)
  #     run_experiment("dump/epsilon-final-choco-top-8bit-" + split_way + "-" + str(cores)+'-ring'\
  #                  + "/", dataset_path, params,n_repeat, nproc=1)
  # if x in ['final']:
  #   params = []
  #   for cores in n_cores:
  #     params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, consensus_lr=0.34,
  #                          quantization='top-qsgd-unbiased', num_levels=256,coordinates_to_keep=1000,
  #                          n_cores=cores, method='choco', topology='star',
  #                          estimate='final', split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name)
  #     run_experiment("dump/epsilon-final-choco-top-8bit-" + split_way + "-" + str(cores)+'-random'\
  #                  + "/", dataset_path, params,n_repeat, nproc=1)
  #
  #
  #
  #
  # if x in ['final']:
  #   params = []
  #   for cores in n_cores:
  #     params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=1000,
  #                          quantization='top-qsgd-unbiased', num_levels=16,
  #                          n_cores=cores, method='choco', topology='star',
  #                          estimate='final', split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name)
  #     run_experiment("dump/epsilon-final-choco-top-4bit-" + split_way + "-" + str(cores)+'-star'\
  #                  + "/", dataset_path, params,n_repeat, nproc=1)
  #
  # if x in ['final']:
  #   params = []
  #   for cores in n_cores:
  #     params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=1000,
  #                          quantization='top-qsgd-unbiased', num_levels=16,
  #                          n_cores=cores, method='choco', topology='star',
  #                          estimate='final', split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name)
  #     run_experiment("dump/epsilon-final-choco-top-4bit-" + split_way + "-" + str(cores)+'-random'\
  #                  + "/", dataset_path, params,n_repeat, nproc=1)
  #
  #
  #
  # if x in ['final']:
  #   params = []
  #   for cores in n_cores:
  #     params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=1000,
  #                          quantization='top-qsgd-unbiased', num_levels=4,
  #                          n_cores=cores, method='choco', topology='star',
  #                          estimate='final', split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name)
  #     run_experiment("dump/epsilon-final-choco-top-2bit-" + split_way + "-" + str(cores)+'-star'\
  #                  + "/", dataset_path, params,n_repeat, nproc=1)
  # if x in ['final']:
  #   params = []
  #   for cores in n_cores:
  #     params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=1000,
  #                          quantization='top-qsgd-unbiased', num_levels=4,
  #                          n_cores=cores, method='choco', topology='ring',
  #                          estimate='final', split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name)
  #     run_experiment("dump/epsilon-final-choco-top-2bit-" + split_way + "-" + str(cores)+'-ring'\
  #                  + "/", dataset_path, params,n_repeat, nproc=1)
  # if x in ['final']:
  #   params = []
  #   for cores in n_cores:
  #     params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
  #                          lr_type='decay', initial_lr=0.1, tau=d,
  #                          regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=1000,
  #                          quantization='top-qsgd-unbiased', num_levels=4,
  #                          n_cores=cores, method='choco', topology='star',
  #                          estimate='final', split_data_random_seed=random_seed,
  #                          distribute_data=True, split_data_strategy=split_name)
  #     run_experiment("dump/epsilon-final-choco-top-2bit-" + split_way + "-" + str(cores)+'-random'\
  #                  + "/", dataset_path, params,n_repeat, nproc=1)
  #
  #









#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=1000,
#                            quantization='top-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='choco', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-4bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#
#
#
#
#
#
# # ajuofkhas;lf
# #
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=400,
#                            quantization='top-qsgd-unbiased', num_levels=256,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-400-8bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=400,
#                            quantization='top-qsgd-unbiased', num_levels=256,
#                            n_cores=cores, method='choco', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-400-8bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,
#                            quantization='top-qsgd-unbiased', num_levels=256,coordinates_to_keep=400,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-400-8bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#
#
#
#
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=400,
#                            quantization='top-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-400-4bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=400,
#                            quantization='top-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='choco', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-400-4bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=400,
#                            quantization='top-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-400-4bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#
#
#
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=400,
#                            quantization='top-qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-400-2bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=400,
#                            quantization='top-qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='choco', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-400-2bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=400,
#                            quantization='top-qsgd-unbiased', num_levels=4,
# #                            n_cores=cores, method='choco', topology='star',
# #                            estimate='final', split_data_random_seed=random_seed,
# #                            distribute_data=True, split_data_strategy=split_name)
# #       run_experiment("dump/epsilon-final-choco-top-400-2bit-" + split_way + "-" + str(cores)+'-random'\
# #                    + "/", dataset_path, params,n_repeat, nproc=1)
# #
#
#
#
# #  CHOCO 200 COORDINATED
#
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=200,
#                            quantization='top-qsgd-unbiased', num_levels=256,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-200-8bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=200,
#                            quantization='top-qsgd-unbiased', num_levels=256,
#                            n_cores=cores, method='choco', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-200-8bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,
#                            quantization='top-qsgd-unbiased', num_levels=256,coordinates_to_keep=200,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-200-8bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#
#
#
#
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=200,
#                            quantization='top-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-200-4bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=200,
#                            quantization='top-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='choco', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-200-4bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=200,
#                            quantization='top-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-200-4bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#
#
#
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=200,
#                            quantization='top-qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-200-2bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=200,
#                            quantization='top-qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='choco', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-200-2bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, consensus_lr=0.34,coordinates_to_keep=200,
#                            quantization='top-qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='choco', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-choco-top-200-2bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
# #
#
#
# # CHOCO RANDOM 200
#
#     if x in ['final']:
#         params = []
#         for cores in n_cores:
#             params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                                 lr_type='decay', initial_lr=0.1, tau=d,
#                                 regularizer=1 / n, consensus_lr=0.34, coordinates_to_keep=200,
#                                 quantization='random-qsgd-unbiased', num_levels=256,
#                                 n_cores=cores, method='choco', topology='star',
#                                 estimate='final', split_data_random_seed=random_seed,
#                                 distribute_data=True, split_data_strategy=split_name)
#             run_experiment("dump/epsilon-final-choco-random-200-8bit-" + split_way + "-" + str(cores) + '-star' \
#                            + "/", dataset_path, params, n_repeat, nproc=1)
#     if x in ['final']:
#         params = []
#         for cores in n_cores:
#             params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                                 lr_type='decay', initial_lr=0.1, tau=d,
#                                 regularizer=1 / n, consensus_lr=0.34, coordinates_to_keep=200,
#                                 quantization='random-qsgd-unbiased', num_levels=256,
#                                 n_cores=cores, method='choco', topology='ring',
#                                 estimate='final', split_data_random_seed=random_seed,
#                                 distribute_data=True, split_data_strategy=split_name)
#             run_experiment("dump/epsilon-final-choco-random-200-8bit-" + split_way + "-" + str(cores) + '-ring' \
#                            + "/", dataset_path, params, n_repeat, nproc=1)
#     if x in ['final']:
#         params = []
#         for cores in n_cores:
#             params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                                 lr_type='decay', initial_lr=0.1, tau=d,
#                                 regularizer=1 / n, consensus_lr=0.34,
#                                 quantization='random-qsgd-unbiased', num_levels=256, coordinates_to_keep=200,
#                                 n_cores=cores, method='choco', topology='star',
#                                 estimate='final', split_data_random_seed=random_seed,
#                                 distribute_data=True, split_data_strategy=split_name)
#             run_experiment("dump/epsilon-final-choco-random-200-8bit-" + split_way + "-" + str(cores) + '-random' \
#                            + "/", dataset_path, params, n_repeat, nproc=1)
#
#     if x in ['final']:
#         params = []
#         for cores in n_cores:
#             params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                                 lr_type='decay', initial_lr=0.1, tau=d,
#                                 regularizer=1 / n, consensus_lr=0.34, coordinates_to_keep=200,
#                                 quantization='random-qsgd-unbiased', num_levels=16,
#                                 n_cores=cores, method='choco', topology='star',
#                                 estimate='final', split_data_random_seed=random_seed,
#                                 distribute_data=True, split_data_strategy=split_name)
#             run_experiment("dump/epsilon-final-choco-random-200-4bit-" + split_way + "-" + str(cores) + '-star' \
#                            + "/", dataset_path, params, n_repeat, nproc=1)
#     if x in ['final']:
#         params = []
#         for cores in n_cores:
#             params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                                 lr_type='decay', initial_lr=0.1, tau=d,
#                                 regularizer=1 / n, consensus_lr=0.34, coordinates_to_keep=200,
#                                 quantization='random-qsgd-unbiased', num_levels=16,
#                                 n_cores=cores, method='choco', topology='ring',
#                                 estimate='final', split_data_random_seed=random_seed,
#                                 distribute_data=True, split_data_strategy=split_name)
#             run_experiment("dump/epsilon-final-choco-random-200-4bit-" + split_way + "-" + str(cores) + '-ring' \
#                            + "/", dataset_path, params, n_repeat, nproc=1)
#     if x in ['final']:
#         params = []
#         for cores in n_cores:
#             params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                                 lr_type='decay', initial_lr=0.1, tau=d,
#                                 regularizer=1 / n, consensus_lr=0.34, coordinates_to_keep=200,
#                                 quantization='random-qsgd-unbiased', num_levels=16,
#                                 n_cores=cores, method='choco', topology='star',
#                                 estimate='final', split_data_random_seed=random_seed,
#                                 distribute_data=True, split_data_strategy=split_name)
#             run_experiment("dump/epsilon-final-choco-random-200-4bit-" + split_way + "-" + str(cores) + '-random' \
#                            + "/", dataset_path, params, n_repeat, nproc=1)
#
#     if x in ['final']:
#         params = []
#         for cores in n_cores:
#             params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                                 lr_type='decay', initial_lr=0.1, tau=d,
#                                 regularizer=1 / n, consensus_lr=0.34, coordinates_to_keep=200,
#                                 quantization='random-qsgd-unbiased', num_levels=4,
#                                 n_cores=cores, method='choco', topology='star',
#                                 estimate='final', split_data_random_seed=random_seed,
#                                 distribute_data=True, split_data_strategy=split_name)
#             run_experiment("dump/epsilon-final-choco-random-200-2bit-" + split_way + "-" + str(cores) + '-star' \
#                            + "/", dataset_path, params, n_repeat, nproc=1)
#     if x in ['final']:
#         params = []
#         for cores in n_cores:
#             params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                                 lr_type='decay', initial_lr=0.1, tau=d,
#                                 regularizer=1 / n, consensus_lr=0.34, coordinates_to_keep=200,
#                                 quantization='random-qsgd-unbiased', num_levels=4,
#                                 n_cores=cores, method='choco', topology='ring',
#                                 estimate='final', split_data_random_seed=random_seed,
#                                 distribute_data=True, split_data_strategy=split_name)
#             run_experiment("dump/epsilon-final-choco-random-200-2bit-" + split_way + "-" + str(cores) + '-ring' \
#                            + "/", dataset_path, params, n_repeat, nproc=1)
#     if x in ['final']:
#         params = []
#         for cores in n_cores:
#             params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                                 lr_type='decay', initial_lr=0.1, tau=d,
#                                 regularizer=1 / n, consensus_lr=0.34, coordinates_to_keep=200,
#                                 quantization='random-qsgd-unbiased', num_levels=4,
#                                 n_cores=cores, method='choco', topology='star',
#                                 estimate='final', split_data_random_seed=random_seed,
#                                 distribute_data=True, split_data_strategy=split_name)
#             run_experiment("dump/epsilon-final-choco-random-200-2bit-" + split_way + "-" + str(cores) + '-random' \
#                            + "/", dataset_path, params, n_repeat, nproc=1)
#   #
# #  choco random 400
#
#   if x in ['final']:
#       params = []
#       for cores in n_cores:
#           params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=0.1, tau=d,
#                               regularizer=1 / n, consensus_lr=0.34, coordinates_to_keep=400,
#                               quantization='random-qsgd-unbiased', num_levels=256,
#                               n_cores=cores, method='choco', topology='star',
#                               estimate='final', split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name)
#           run_experiment("dump/epsilon-final-choco-random-400-8bit-" + split_way + "-" + str(cores) + '-star' \
#                          + "/", dataset_path, params, n_repeat, nproc=1)
#   if x in ['final']:
#       params = []
#       for cores in n_cores:
#           params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=0.1, tau=d,
#                               regularizer=1 / n, consensus_lr=0.34, coordinates_to_keep=400,
#                               quantization='random-qsgd-unbiased', num_levels=256,
#                               n_cores=cores, method='choco', topology='ring',
#                               estimate='final', split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name)
#           run_experiment("dump/epsilon-final-choco-random-400-8bit-" + split_way + "-" + str(cores) + '-ring' \
#                          + "/", dataset_path, params, n_repeat, nproc=1)
#   if x in ['final']:
#       params = []
#       for cores in n_cores:
#           params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=0.1, tau=d,
#                               regularizer=1 / n, consensus_lr=0.34,
#                               quantization='random-qsgd-unbiased', num_levels=256, coordinates_to_keep=400,
#                               n_cores=cores, method='choco', topology='star',
#                               estimate='final', split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name)
#           run_experiment("dump/epsilon-final-choco-random-400-8bit-" + split_way + "-" + str(cores) + '-random' \
#                          + "/", dataset_path, params, n_repeat, nproc=1)
#
#   if x in ['final']:
#       params = []
#       for cores in n_cores:
#           params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=0.1, tau=d,
#                               regularizer=1 / n, consensus_lr=0.34, coordinates_to_keep=400,
#                               quantization='random-qsgd-unbiased', num_levels=16,
#                               n_cores=cores, method='choco', topology='star',
#                               estimate='final', split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name)
#           run_experiment("dump/epsilon-final-choco-random-400-4bit-" + split_way + "-" + str(cores) + '-star' \
#                          + "/", dataset_path, params, n_repeat, nproc=1)
#   if x in ['final']:
#       params = []
#       for cores in n_cores:
#           params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=0.1, tau=d,
#                               regularizer=1 / n, consensus_lr=0.34, coordinates_to_keep=400,
#                               quantization='random-qsgd-unbiased', num_levels=16,
#                               n_cores=cores, method='choco', topology='ring',
#                               estimate='final', split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name)
#           run_experiment("dump/epsilon-final-choco-random-400-4bit-" + split_way + "-" + str(cores) + '-ring' \
#                          + "/", dataset_path, params, n_repeat, nproc=1)
#   if x in ['final']:
#       params = []
#       for cores in n_cores:
#           params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=0.1, tau=d,
#                               regularizer=1 / n, consensus_lr=0.34, coordinates_to_keep=400,
#                               quantization='random-qsgd-unbiased', num_levels=16,
#                               n_cores=cores, method='choco', topology='star',
#                               estimate='final', split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name)
#           run_experiment("dump/epsilon-final-choco-random-400-4bit-" + split_way + "-" + str(cores) + '-random' \
#                          + "/", dataset_path, params, n_repeat, nproc=1)
#
#   if x in ['final']:
#       params = []
#       for cores in n_cores:
#           params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=0.1, tau=d,
#                               regularizer=1 / n, consensus_lr=0.34, coordinates_to_keep=400,
#                               quantization='random-qsgd-unbiased', num_levels=4,
#                               n_cores=cores, method='choco', topology='star',
#                               estimate='final', split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name)
#           run_experiment("dump/epsilon-final-choco-random-400-2bit-" + split_way + "-" + str(cores) + '-star' \
#                          + "/", dataset_path, params, n_repeat, nproc=1)
#   if x in ['final']:
#       params = []
#       for cores in n_cores:
#           params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=0.1, tau=d,
#                               regularizer=1 / n, consensus_lr=0.34, coordinates_to_keep=400,
#                               quantization='random-qsgd-unbiased', num_levels=4,
#                               n_cores=cores, method='choco', topology='ring',
#                               estimate='final', split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name)
#           run_experiment("dump/epsilon-final-choco-random-400-2bit-" + split_way + "-" + str(cores) + '-ring' \
#                          + "/", dataset_path, params, n_repeat, nproc=1)
#   if x in ['final']:
#       params = []
#       for cores in n_cores:
#           params = Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                               lr_type='decay', initial_lr=0.1, tau=d,
#                               regularizer=1 / n, consensus_lr=0.34, coordinates_to_keep=400,
#                               quantization='random-qsgd-unbiased', num_levels=4,
#                               n_cores=cores, method='choco', topology='star',
#                               estimate='final', split_data_random_seed=random_seed,
#                               distribute_data=True, split_data_strategy=split_name)
#           run_experiment("dump/epsilon-final-choco-random-400-2bit-" + split_way + "-" + str(cores) + '-random' \
#                          + "/", dataset_path, params, n_repeat, nproc=1)



# DCD TOP 200

#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=200,
#                            quantization='top-qsgd-unbiased', num_levels=256,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-200-8bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n,coordinates_to_keep=200,
#                            quantization='top-qsgd-unbiased', num_levels=256,
#                            n_cores=cores, method='dcd-psgd', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-200-8bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n,
#                            quantization='top-qsgd-unbiased', num_levels=256,coordinates_to_keep=200,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-200-8bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#
#
#
#
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=200,
#                            quantization='top-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-200-4bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=200,
#                            quantization='top-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='dcd-psgd', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-200-4bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n,coordinates_to_keep=200,
#                            quantization='top-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-200-4bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#
#
#
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=200,
#                            quantization='top-qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-200-2bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=200,
#                            quantization='top-qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='dcd-psgd', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-200-2bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=200,
#                            quantization='top-qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-200-2bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
# #



# dcd top 400
#
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n,coordinates_to_keep=400,
#                            quantization='top-qsgd-unbiased', num_levels=256,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-400-8bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n,coordinates_to_keep=400,
#                            quantization='top-qsgd-unbiased', num_levels=256,
#                            n_cores=cores, method='dcd-psgd', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-400-8bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n,
#                            quantization='top-qsgd-unbiased', num_levels=256,coordinates_to_keep=400,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-400-8bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#
#
#
#
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=400,
#                            quantization='top-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-400-4bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n,coordinates_to_keep=400,
#                            quantization='top-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='dcd-psgd', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-400-4bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=400,
#                            quantization='top-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-400-4bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#
#
#
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n,coordinates_to_keep=400,
#                            quantization='top-qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-400-2bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n,coordinates_to_keep=400,
#                            quantization='top-qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='dcd-psgd', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-400-2bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=400,
#                            quantization='top-qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-top-400-2bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#





# DCD RANDOM 400



#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=400,
#                            quantization='random-qsgd-unbiased', num_levels=256,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-400-8bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=400,
#                            quantization='random-qsgd-unbiased', num_levels=256,
#                            n_cores=cores, method='dcd-psgd', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-400-8bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n,
#                            quantization='random-qsgd-unbiased', num_levels=256,coordinates_to_keep=400,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-400-8bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#
#
#
#
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n,coordinates_to_keep=400,
#                            quantization='random-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-400-4bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=400,
#                            quantization='random-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='dcd-psgd', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-400-4bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=400,
#                            quantization='random-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-400-4bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#
#
#
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=400,
#                            quantization='random-qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-400-2bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=400,
#                            quantization='random-qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='dcd-psgd', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-400-2bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n,coordinates_to_keep=400,
#                            quantization='random-qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-400-2bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#
# # DCD RANDOM 200
#
#
#
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n,coordinates_to_keep=200,
#                            quantization='random-qsgd-unbiased', num_levels=256,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-200-8bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=200,
#                            quantization='random-qsgd-unbiased', num_levels=256,
#                            n_cores=cores, method='dcd-psgd', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-200-8bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n,
#                            quantization='random-qsgd-unbiased', num_levels=256,coordinates_to_keep=200,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-200-8bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#
#
#
#
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n,coordinates_to_keep=200,
#                            quantization='random-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-200-4bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n,coordinates_to_keep=200,
#                            quantization='random-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='dcd-psgd', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-200-4bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n,coordinates_to_keep=200,
#                            quantization='random-qsgd-unbiased', num_levels=16,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-200-4bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#
#
#
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=200,
#                            quantization='random-qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-200-2bit-" + split_way + "-" + str(cores)+'-star'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=200,
#                            quantization='random-qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='dcd-psgd', topology='ring',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-200-2bit-" + split_way + "-" + str(cores)+'-ring'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)
#   if x in ['final']:
#     params = []
#     for cores in n_cores:
#       params =Parameters(name="decentralized-qsgd-top", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, coordinates_to_keep=200,
#                            quantization='random-qsgd-unbiased', num_levels=4,
#                            n_cores=cores, method='dcd-psgd', topology='star',
#                            estimate='final', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name)
#       run_experiment("dump/epsilon-final-dcd-random-200-2bit-" + split_way + "-" + str(cores)+'-random'\
#                    + "/", dataset_path, params,n_repeat, nproc=1)


###############################################
### SORTED DATA PARTITION #####################
###############################################
################################ FINAL ####################################




#   split_way = 'sorted'
#   split_name = split_way
#   if split_way == 'sorted':
#     split_name = 'label-sorted'
#
#   n_repeat = 5
#   num_epoch = 10
#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [
#                 Parameters(name="decentralized-exact", num_epoch=num_epoch,
#                            lr_type='decay', initial_lr=0.1, tau=d,
#                            regularizer=1 / n, quantization='full',
#                            n_cores=n_cores, method='plain',
#                            split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name,
#                            topology='ring', estimate='final'),
#       ]
#     run_experiment("dump/epsilon-final-decentralized-" + split_way+ "-" + str(n_cores)\
#                    + "/", dataset_path, params, nproc=10)
#
#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [
#                 Parameters(name="centralized", num_epoch=num_epoch, lr_type='decay',
#                            initial_lr=0.1, tau=d, regularizer=1 / n,
#                            quantization='full', n_cores=n_cores, method='plain',
#                            split_data_random_seed=random_seed, distribute_data=True,
#                            split_data_strategy=split_name, topology='centralized',
#                            estimate='final')]
#     run_experiment("dump/epsilon-final-centralized-" + split_way+ "-" + str(n_cores)\
#                    + "/", dataset_path, params, nproc=10)
#
#
#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [
#                 Parameters(name="decentralized-top-20", num_epoch=num_epoch, lr_type='decay',
#                            initial_lr=0.1, tau=d, regularizer=1 / n, consensus_lr=0.04,
#                            quantization='top', coordinates_to_keep=20, n_cores=n_cores,
#                            method='choco', topology='ring', estimate='final',
#                            split_data_random_seed=random_seed, distribute_data=True,
#                            split_data_strategy=split_name, random_seed=40 + random_seed)]
#     run_experiment("dump/epsilon-final-choco-top-20-" + split_way+ "-" + str(n_cores) + "/",
#         dataset_path, params, nproc=10)
#
#
#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [
#                 Parameters(name="decentralized-random-20", num_epoch=num_epoch, lr_type='decay',
#                            initial_lr=0.1, tau=d, regularizer=1 / n, consensus_lr=0.01,
#                            quantization='random-unbiased, coordinates_to_keep=20, n_cores=n_cores,
#                            method='choco', topology='ring', estimate='final',
#                            split_data_random_seed=random_seed, distribute_data=True,
#                            split_data_strategy=split_name, random_seed=60 + random_seed)]
#     run_experiment("dump/epsilon-final-choco-random-20-" + split_way+ "-" + str(n_cores) + "/",
#           dataset_path, params, nproc=10)
#
#
#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [
#                 Parameters(name="decentralized-qsgd-8", num_epoch=num_epoch, lr_type='decay',
#                            initial_lr=0.1, tau=d, regularizer=1 / n, consensus_lr=0.34,
#                            quantization='qsgd-unbiased, num_levels=16, n_cores=n_cores,
#                            method='choco', topology='ring', estimate='final',
#                            split_data_random_seed=random_seed, distribute_data=True,
#                            split_data_strategy=split_name)]
#     run_experiment("dump/epsilon-final-choco-qsgd-4bit-" + split_way + "-" + str(n_cores) + "/",
#                    dataset_path, params, nproc=10)
#
#
#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [Parameters(name="dcd-psgd-random-20", num_epoch=num_epoch,
#                             lr_type='decay', initial_lr=1e-15, tau=d,
#                             regularizer=1 / n, quantization='random-unbiased',
#                             coordinates_to_keep=20, n_cores=n_cores, method='dcd-psgd',
#                             split_data_random_seed=random_seed, distribute_data=True,
#                             split_data_strategy=split_name, topology='ring',
#                             estimate='final')]
#     run_experiment("dump/epsilon-final-dcd-random-20-" + split_way + "-" + str(n_cores) + "/",
#                    dataset_path, params, nproc=10)
#
#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [Parameters(name="ecd-psgd-random",
#                            num_epoch=num_epoch, lr_type='decay',
#                            initial_lr=1e-10, tau=d, regularizer=1 / n,
#                            quantization='random-unbiased', coordinates_to_keep=20,
#                            n_cores=n_cores,
#                            method='ecd-psgd', split_data_random_seed=random_seed,
#                            distribute_data=True,
#                            split_data_strategy=split_name,
#                            topology='ring', estimate='final')]
#     run_experiment("dump/epsilon-final-ecd-random-20-" + split_way + "-" + str(n_cores) + "/",
#                    dataset_path, params, nproc=10)
#
# ###################### qsgd quantization #####################################
#
#
#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [Parameters(name="dcd-psgd-qsgd",
#                            num_epoch=num_epoch, lr_type='decay',
#                            initial_lr=0.01, tau=d, regularizer=1 / n,
#                            quantization='qsgd-unbiased', num_levels=16, n_cores=n_cores,
#                            method='dcd-psgd', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name,
#                            topology='ring', estimate='final')]
#     run_experiment("dump/epsilon-final-dcd-qsgd-4bit-" + split_way + "-" + str(n_cores) + "/",
#                    dataset_path, params, nproc=10)
#
#   if args.experiment in ['final']:
#     params = []
#     for random_seed in np.arange(1, n_repeat + 1):
#       params += [Parameters(name="ecd-psgd-qsgd",
#                            num_epoch=num_epoch, lr_type='decay',
#                            initial_lr=1e-12, tau=d, regularizer=1 / n,
#                            quantization='qsgd-unbiased', num_levels=16, n_cores=n_cores,
#                            method='ecd-psgd', split_data_random_seed=random_seed,
#                            distribute_data=True, split_data_strategy=split_name,
#                            topology='ring', estimate='final')]
#     run_experiment("dump/epsilon-final-ecd-qsgd-4bit-" + split_way + "-" + str(n_cores) + "/",
#                    dataset_path, params, nproc=10)
#
#
