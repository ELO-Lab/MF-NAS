import argparse
import logging
import sys
import numpy as np
from datetime import datetime
from factory import get_problem, get_algorithm
import json
import pickle as p
import os
import pathlib

root = pathlib.Path(__file__).parent

def run(kwargs):
    dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    opt_name = kwargs.optimizer

    search_space = kwargs.ss
    dataset = kwargs.dataset
    if '201' in search_space:
        search_space += f'_{dataset}'
    res_path = f'{root}/exp/{opt_name}_{search_space}_' + dt_string
    os.makedirs(res_path, exist_ok=True)
    os.makedirs(res_path + '/results', exist_ok=True)

    problem, info_problem = get_problem(search_space, res_path=res_path + '/results')

    config_file = kwargs.config_file
    opt, info_algo = get_algorithm(opt_name, config_file)
    opt.adapt(problem)

    n_run = kwargs.n_run
    trend_search_cost, trend_total_epoch = [], []

    os.makedirs(res_path + '/configs', exist_ok=True)
    json.dump(info_problem, open(res_path + '/configs/info_problem.json', 'w'), indent=4)
    json.dump(info_algo, open(res_path + '/configs/info_algo.json', 'w'), indent=4)

    init_seed = args.init_seed
    for run_id in range(1, n_run + 1):
        search_result, search_cost, total_epoch = opt.run(seed=init_seed + run_id)
        p.dump([search_result, int(search_cost), int(total_epoch)], open(res_path + f'/results/run_{run_id}_results.p', 'wb'))
        print(f'- RunID: {run_id}')
        print(f'  + Search cost (in seconds): {int(search_cost)}')
        print(f'  + Search cost (in epochs): {int(total_epoch)}')
        print('-' * 100, '\n')
        trend_search_cost.append(search_cost)
        trend_total_epoch.append(total_epoch)
    print(f'- Average Search Cost (in seconds): {np.round(np.mean(trend_search_cost))}')
    print(f'- Average Search Cost (in epochs): {np.round(np.mean(trend_total_epoch))}')
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
    choices=['RS', 'SH', 'FLS', 'BLS', 'REA', 'REA+W', 'MF-NAS', 'PLS', 'NSGA2'])
    parser.add_argument('--config_file', type=str, default='./configs/algo_201.yaml', help='the configuration file')

    ''' ENVIRONMENT '''
    parser.add_argument('--n_run', type=int, default=500, help='number of experiment runs')
    parser.add_argument('--init_seed', type=int, default=0, help='initial random seed')

    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    run(args)