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
from utils import create_result_folder, mean_std

root = str(pathlib.Path.cwd())
log = True

def run(kwargs):
    opt_name = kwargs.optimizer
    config_file = kwargs.config_file
    opt, info_algo = get_algorithm(opt_name, config_file)

    search_space = kwargs.ss
    dataset = kwargs.dataset
    if search_space in ['nb201', 'nats']:
        search_space += f'_{dataset}'

    save_path = kwargs.save_path
    res_path = create_result_folder(opt.nas_type, opt_name, search_space, opt, save_path)

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
    list_perf_ind = []
    from tqdm import tqdm
    for run_id in tqdm(range(1, n_run + 1)):
        search_result, search_cost, total_epoch = opt.run(seed=init_seed + run_id)
        opt.finalize(save_path=res_path + f'/results', rid=run_id)

        p.dump([search_result, int(search_cost), int(total_epoch)], open(res_path + f'/results/search_results_run{run_id}.p', 'wb'))
        trend_search_cost.append(search_cost)
        trend_total_epoch.append(total_epoch)

        if kwargs.evaluate_after_search:
            size_archive = 20
            evaluation_result = run_evaluate(
                search_result=search_result, search_cost=search_cost, total_epoch=total_epoch,
                problem=problem, nas_type=opt.nas_type,
                save_path=res_path + f'/results',
                filename=f'evaluation_results_run{run_id}', size_archive=size_archive, verbose=False, log=log)
            list_perf_ind.append(evaluation_result['Networks'][-1]['test_acc'])
    print('- Average performance:', mean_std(list_perf_ind, verbose=False))
    print(f'- Average Search Cost (in seconds): {np.round(np.mean(trend_search_cost))}')
    print(f'- Average Search Cost (in epochs): {np.round(np.mean(trend_total_epoch))}')
    print('=' * 100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' PROBLEM '''
    parser.add_argument('--ss', type=str, default='nb201', help='the search space',
    choices=['nb201', 'nb101', 'nbasr', 'darts', 'nats'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'],
    help='dataset for NAS-Bench-201')

    ''' ALGORITHM '''
    parser.add_argument('--optimizer', type=str, default='MF-NAS', help='the search strategy')
    parser.add_argument('--config_file', type=str, default='./configs/algo_201.yaml', help='the configuration file')

    ''' ENVIRONMENT '''
    parser.add_argument('--n_run', type=int, default=500, help='number of experiment runs')
    parser.add_argument('--init_seed', type=int, default=0, help='initial random seed')
    parser.add_argument('--evaluate_after_search', help='Evaluate the results after searching', action='store_true')
    parser.add_argument('--save_path', default=f'{root}/exp')

    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    run(args)