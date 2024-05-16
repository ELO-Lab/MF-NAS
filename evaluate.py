from models import Network
from algos.utils import ElitistArchive

import argparse
import logging
import sys
import numpy as np
from factory import get_problem, get_algorithm
import json
import pickle as p
import os
import pathlib
root = pathlib.Path(__file__).parent

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def so_evaluate(search_result: Network, problem):
    evaluation_result = []
    test_performance = problem.get_test_performance(search_result)
    evaluation_result.append({'genotype': search_result.genotype,
                              'phenotype': problem.search_space.decode(search_result.genotype),
                              'test_acc': test_performance})
    return evaluation_result

def mo_evaluate(search_result: ElitistArchive, problem, algo):
    evaluation_result = []
    list_networks = search_result.archive
    list_metrics = algo.list_metrics[1:]
    for network_id, network in enumerate(list_networks):
        evaluation_result.append({'genotype': network.genotype, 'phenotype': problem.search_space.decode(network.genotype)})
        # scores = {}
        # scores['test_acc'] = test_performance
        evaluation_result[-1]['test_acc'] = problem.get_test_performance(network)
        network_scores = network.score[1:]
        for i, metric in enumerate(list_metrics):
            evaluation_result[-1][metric] = network_scores[i]
    return evaluation_result

def run(kwargs):
    nas_type = kwargs.nas_type
    opt_name = kwargs.optimizer
    res_path = kwargs.res_path

    res_files = os.listdir(kwargs.res_path + '/results')

    search_space = kwargs.ss
    dataset = kwargs.dataset
    if '201' in search_space:
        search_space += f'_{dataset}'

    problem, _ = get_problem(search_space)
    config_file = kwargs.config_file
    opt, _ = get_algorithm(opt_name, config_file)

    result = {}
    print('Results Path:', res_path)
    for run_id, res_file in enumerate(res_files):
        print(f'- RunID: {run_id}\n')
        _res_path = res_path + '/results/' + res_file
        search_result, search_cost, total_epoch = p.load(open(_res_path, 'rb'))
        if nas_type == 'so':
            evaluation_result = so_evaluate(search_result, problem)
        else:
            evaluation_result = mo_evaluate(search_result, problem, opt)
        for network_id, info in enumerate(evaluation_result):
            print(f'   * Network #{network_id}:')
            for key in info:
                print(f'     + {key}: {info[key]}')
            print()
        print('-' * 100)

        result[f'{run_id}'] = {'info': evaluation_result,
                               'Search cost (in seconds)': search_cost,
                               'Search epoch (in seconds)': total_epoch}

    with open(f'{res_path}/results.json', 'w') as fp:
        json.dump(result, fp, indent=4, cls=NumpyEncoder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' PROBLEM '''
    parser.add_argument('--nas_type', type=str, default='so', help='the problem type', choices=['so', 'mo'])
    parser.add_argument('--res_path', type=str)
    parser.add_argument('--ss', type=str, default='nb201', help='the search space',
    choices=['nb201', 'nb101', 'nbasr', 'darts'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'],
    help='dataset for NAS-Bench-201')

    parser.add_argument('--optimizer', type=str, default='MF-NAS', help='the search strategy',
    choices=['RS', 'SH', 'FLS', 'BLS', 'REA', 'REA+W', 'MF-NAS', 'PLS', 'NSGA2'])
    parser.add_argument('--config_file', type=str, default='./configs/algo_201.yaml', help='the configuration file')

    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    run(args)