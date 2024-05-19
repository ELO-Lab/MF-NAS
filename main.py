import argparse
import logging
import sys
import numpy as np
from factory import get_problem, get_algorithm
import json
import pickle as p
import os
import pathlib
from evaluate import run_evaluate
from utils import create_result_folder

root = pathlib.Path(__file__).parent

def run(kwargs):
    opt_name = kwargs.optimizer
    config_file = kwargs.config_file
    opt, info_algo = get_algorithm(opt_name, config_file)

    search_space = kwargs.ss
    dataset = kwargs.dataset
    if '201' in search_space:
        search_space += f'_{dataset}'

    res_path = create_result_folder(opt.nas_type, root, opt_name, search_space, opt)

    os.makedirs(res_path + '/results', exist_ok=True)

    problem, info_problem = get_problem(search_space, res_path=res_path + '/results')

    opt.adapt(problem)

    n_run = kwargs.n_run
    trend_search_cost, trend_total_epoch = [], []

    os.makedirs(res_path + '/configs', exist_ok=True)
    json.dump(info_problem, open(res_path + '/configs/info_problem.json', 'w'), indent=4)
    json.dump(info_algo, open(res_path + '/configs/info_algo.json', 'w'), indent=4)

    init_seed = args.init_seed
    print()
    for run_id in [3]:
        search_result, search_cost, total_epoch = opt.run(seed=init_seed + run_id)
        p.dump([search_result, int(search_cost), int(total_epoch)], open(res_path + f'/results/search_results_run{run_id}.p', 'wb'))
        print(f'- RunID: {run_id}')
        print(f'  + Search cost (in seconds): {int(search_cost)}')
        print(f'  + Search cost (in epochs): {int(total_epoch)}')
        print('-'*100)
        trend_search_cost.append(search_cost)
        trend_total_epoch.append(total_epoch)

        if kwargs.evaluate_after_search:
            run_evaluate(res_path + f'/results/search_results_run{run_id}.p', problem, opt.nas_type, save_path=res_path + f'/results',
            filename=f'evaluation_results_run{run_id}', size_archive=20)
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
    parser.add_argument('--optimizer', type=str, default='MF-NAS', help='the search strategy')
    parser.add_argument('--config_file', type=str, default='./configs/algo_201.yaml', help='the configuration file')

    ''' ENVIRONMENT '''
    parser.add_argument('--n_run', type=int, default=500, help='number of experiment runs')
    parser.add_argument('--init_seed', type=int, default=0, help='initial random seed')
    parser.add_argument('--evaluate_after_search', help='Evaluate the results after searching', action='store_true')

    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    run(args)