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
    search_space = kwargs.ss
    dataset = kwargs.dataset
    if '201' in search_space:
        search_space += f'_{dataset}'
    problem, info_problem = get_problem(search_space)

    opt_name = kwargs.optimizer
    opt, info_algo = get_algorithm(opt_name)
    opt.adapt(problem)

    n_run = kwargs.n_run
    verbose = kwargs.verbose
    trend_performance, trend_search_cost, trend_total_epoch = [], [], []

    dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    root = './exp' if kwargs.path_res is None else kwargs.path_res
    path_res = f'{root}/{opt_name}_{search_space}_' + dt_string
    try:
        os.mkdir(path_res)
        os.mkdir(path_res + '/configs')
        os.mkdir(path_res + '/results')
    except FileExistsError:
        pass
    json.dump(info_problem, open(path_res + '/configs/info_problem.json', 'w'), indent=4)
    json.dump(info_algo, open(path_res + '/configs/info_algo.json', 'w'), indent=4)

    best_score = []
    all_evaluation_cost = []
    for run_id in tqdm(range(1, n_run + 1)):
        best_network, search_cost, total_epoch = opt.run(seed=run_id)
        best_score.append(best_network.score)

        test_performance, evaluation_cost = problem.get_test_performance(best_network)
        all_evaluation_cost.append(evaluation_cost)

        if verbose:
            network_phenotype = problem.search_space.decode(best_network.genotype)
            print()
            print(f'RunID: {run_id}\n')
            logging.info(f'Best architecture found:\n{network_phenotype}\n')
            logging.info(f'Performance: {test_performance} %')
            logging.info(f'Search cost (in seconds): {search_cost}')
            logging.info(f'Search cost (in epochs): {total_epoch}\n')
            logging.info(f'Evaluation cost (in seconds): {evaluation_cost}\n')
            print('-' * 100)
        trend_performance.append(test_performance)
        trend_search_cost.append(search_cost)
        trend_total_epoch.append(total_epoch)
        info_results = {
            'Genotype': best_network.genotype,
            'Phenotype': problem.search_space.decode(best_network.genotype),
            'Performance': test_performance,
            'Search cost (in seconds)': search_cost,
            'Search cost (in epochs)': total_epoch,
            'Evaluation cost (in seconds)': evaluation_cost,
        }
        p.dump(info_results, open(path_res + f'/results/run_{run_id}_results.p', 'wb'))
        p.dump(opt.search_log, open(path_res + f'/results/log_{run_id}.p', 'wb'))
    logging.info(f'Mean (search metric): {np.round(np.mean(best_score), 4)} \t Std (search metric): {np.round(np.std(best_score), 4)}')
    logging.info(f'Mean: {np.round(np.mean(trend_performance), 2)} \t Std: {np.round(np.std(trend_performance), 2)}')
    logging.info(f'Search cost (in seconds): {np.round(np.mean(trend_search_cost))}')
    logging.info(f'Search cost (in epochs): {np.round(np.mean(trend_total_epoch))}')
    logging.info(f'Evaluation cost (in seconds): {np.round(np.mean(all_evaluation_cost))}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' PROBLEM '''
    parser.add_argument('--ss', type=str, default='nb201', help='the search space', choices=['nb201', 'nb101', 'nbasr'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'],
                        help='dataset for NAS-Bench-201')

    ''' ALGORITHM '''
    parser.add_argument('--optimizer', type=str, default='MF-NAS', help='the search strategy',
                        choices=['RS', 'SH', 'FLS', 'BLS', 'REA', 'GA', 'REA+W', 'MF-NAS'])

    ''' ENVIRONMENT '''
    parser.add_argument('--n_run', type=int, default=31, help='number of experiment runs')
    parser.add_argument('--path_res', type=str, default=None, help='path for saving experiment results')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    run(args)