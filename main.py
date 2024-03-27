import argparse
import logging
import sys
import numpy as np
from tqdm import tqdm
from datetime import datetime
from factory import get_problem, get_algorithm
import json
import pickle as p
import os


def run(kwargs):
    dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    opt_name = kwargs.optimizer

    search_space = kwargs.ss
    dataset = kwargs.dataset
    if '201' in search_space:
        search_space += f'_{dataset}'
    res_path = f'./exp/{opt_name}_{search_space}_' + dt_string
    os.mkdir(res_path)
    os.mkdir(res_path + '/results')

    problem, info_problem = get_problem(search_space, res_path=res_path + '/results')

    config_file = kwargs.config_file
    opt, info_algo = get_algorithm(opt_name, config_file)
    opt.adapt(problem)

    n_run = kwargs.n_run
    verbose = kwargs.verbose
    trend_performance, trend_search_cost, trend_total_epoch = [], [], []

    os.mkdir(res_path + '/configs')
    json.dump(info_problem, open(res_path + '/configs/info_problem.json', 'w'), indent=4)
    json.dump(info_algo, open(res_path + '/configs/info_algo.json', 'w'), indent=4)

    for run_id in tqdm(range(1, n_run + 1)):
        network, search_cost, total_epoch = opt.run(seed=run_id)
        test_performance = problem.get_test_performance(network)
        if verbose:
            network_phenotype = problem.search_space.decode(network.genotype)
            print()
            print(f'RunID: {run_id}\n')
            logging.info(f'Best architecture found:\n{network_phenotype}\n')
            logging.info(f'Performance: {test_performance} %')
            logging.info(f'Search cost (in seconds): {search_cost}')
            logging.info(f'Search cost (in epochs): {total_epoch}\n')
            print('-'*100)
        trend_performance.append(test_performance)
        trend_search_cost.append(search_cost)
        trend_total_epoch.append(total_epoch)
        info_results = {
            'Genotype': network.genotype,
            'Phenotype': problem.search_space.decode(network.genotype),
            'Performance': test_performance,
            'Search cost (in seconds)': search_cost,
            'Search cost (in epochs)': total_epoch,
        }
        p.dump(info_results, open(res_path + f'/results/run_{run_id}_results.p', 'wb'))
    logging.info(f'Mean: {np.round(np.mean(trend_performance), 2)} \t Std: {np.round(np.std(trend_performance), 2)}')
    logging.info(f'Search cost (in seconds): {np.round(np.mean(trend_search_cost))}')
    logging.info(f'Search cost (in epochs): {np.round(np.mean(trend_total_epoch))}')
    print('=' * 100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' PROBLEM '''
    parser.add_argument('--ss', type=str, default='nb201', help='the search space',
    choices=['nb201', 'nb101', 'nbasr', 'darts'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'],
    help='dataset for NAS-Bench-201')

    ''' ALGORITHM '''
    parser.add_argument('--optimizer', type=str, default='MF-NAS', help='the search strategy',
    choices=['RS', 'SH', 'FLS', 'BLS', 'REA', 'REA+W', 'MF-NAS'])
    parser.add_argument('--config_file', type=str, default='./configs/algo_201.yaml', help='the configuration file')

    ''' ENVIRONMENT '''
    parser.add_argument('--n_run', type=int, default=500, help='number of experiment runs')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    run(args)