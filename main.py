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
from utils import so_evaluation_phase, mo_evaluation_phase, print_info

def run(kwargs):
    search_space = kwargs.ss
    dataset = kwargs.dataset
    if '201' in search_space:
        search_space += f'_{dataset}'
    problem, info_problem = get_problem(search_space, max_eval=kwargs.max_eval, max_time=kwargs.max_time)

    opt_name = kwargs.optimizer
    opt, info_algo, multi_objective = get_algorithm(opt_name)
    opt.adapt(problem)

    n_run = kwargs.n_run
    verbose = kwargs.verbose

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

    evaluation_cost_each_run = []
    test_performance_each_run = []
    search_cost_each_run, total_epoch_each_run = [], []

    ### Delete late
    best_score = []
    test_performance_each_run1, evaluation_cost_each_run1 = [], []
    test_performance_each_run2, evaluation_cost_each_run2 = [], []

    for run_id in tqdm(range(1, n_run + 1)):
        search_result, search_cost, total_epoch = opt.run(seed=run_id)
        if multi_objective:
            best_network_1, test_performance_1, evaluation_cost_1, best_network_2, test_performance_2, evaluation_cost_2, test_performances = mo_evaluation_phase(search_result, problem)

            best_score.append(best_network_1.score[0])

            test_performance_each_run1.append(test_performance_1)
            test_performance_each_run2.append(test_performance_2)

            evaluation_cost_each_run1.append(evaluation_cost_1)
            evaluation_cost_each_run2.append(evaluation_cost_2)
            list_genotypes = [problem.search_space.decode(network.genotype) for network in search_result]
            info_results = {
                'Phenotype (1)': problem.search_space.decode(best_network_1.genotype),
                'Performance (1)': test_performance_1,
                'Evaluation cost (1) (in seconds)': evaluation_cost_1,
                'Phenotype (2)': problem.search_space.decode(best_network_2.genotype),
                'Performance (2)': test_performance_2,
                'Evaluation cost (2) (in seconds)': evaluation_cost_2,
                'All networks': list_genotypes,
                'Test performance (all)': test_performances,
                'Search cost (in seconds)': search_cost,
                'Search cost (in epochs)': total_epoch,
            }
            if verbose:
                print(f'RunID: {run_id}\n')
                print_info(info_results)
                print('-' * 100)
            info_results['Genotype (1)'] = best_network_1.genotype
            info_results['Genotype (2)'] = best_network_2.genotype
        else:
            best_network, test_performance, evaluation_cost = so_evaluation_phase(search_result, problem)
            best_score.append(best_network.score)
            evaluation_cost_each_run.append(evaluation_cost)
            test_performance_each_run.append(test_performance)
            info_results = {
                'Phenotype': problem.search_space.decode(best_network.genotype),
                'Performance': test_performance,
                'Search cost (in seconds)': search_cost,
                'Search cost (in epochs)': total_epoch,
                'Evaluation cost (in seconds)': evaluation_cost,
            }
            if verbose:
                print(f'RunID: {run_id}\n')
                print_info(info_results)
                print('-' * 100)
            info_results['Genotype'] = best_network.genotype
        search_cost_each_run.append(search_cost)
        total_epoch_each_run.append(total_epoch)

        p.dump(info_results, open(path_res + f'/results/run_{run_id}_results.p', 'wb'))
        # p.dump(opt.search_log, open(path_res + f'/results/log_{run_id}.p', 'wb'))
    logging.info(
        f'Mean (search): {np.round(np.mean(best_score), 4)} \t Std (search): {np.round(np.std(best_score), 4)}')
    if multi_objective:
        logging.info(f'Mean (1): {np.round(np.mean(test_performance_each_run1), 2)} \t Std: {np.round(np.std(test_performance_each_run1), 2)}')
        logging.info(f'Mean (2): {np.round(np.mean(test_performance_each_run2), 2)} \t Std: {np.round(np.std(test_performance_each_run2), 2)}')
    else:
        logging.info(f'Mean: {np.round(np.mean(test_performance_each_run), 2)} \t Std: {np.round(np.std(test_performance_each_run), 2)}')
    logging.info(f'Search cost: {int(np.mean(search_cost_each_run))} seconds')
    logging.info(f'Search cost: {int(np.mean(total_epoch_each_run))} epochs')

    if multi_objective:
        logging.info(f'Evaluation cost (1): {int(np.mean(evaluation_cost_each_run1))} seconds')
        logging.info(f'Evaluation cost (2): {int(np.mean(evaluation_cost_each_run2))} seconds')
    else:
        logging.info(f'Evaluation cost: {int(np.mean(evaluation_cost_each_run))} seconds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' PROBLEM '''
    parser.add_argument('--ss', type=str, default='nb201', help='the search space', choices=['nb201', 'nb101', 'nbasr'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'],
                        help='dataset for NAS-Bench-201')
    parser.add_argument('--max_eval', type=int, default=None, help='maximum number of evaluations')
    parser.add_argument('--max_time', type=int, default=None, help='maximum times (in seconds)')

    ''' ALGORITHM '''
    parser.add_argument('--optimizer', type=str, default='MF-NAS', help='the search strategy',
                        choices=['RS', 'SH', 'FLS', 'BLS', 'REA', 'GA', 'REA+W', 'MF-NAS', 'MF-GA', 'LOMONAS', 'NSGA2'])

    ''' ENVIRONMENT '''
    parser.add_argument('--n_run', type=int, default=31, help='number of experiment runs')
    parser.add_argument('--path_res', type=str, default=None, help='path for saving experiment results')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    run(args)