from models import Network
from algos.utils import ElitistArchive

import argparse
import logging
import sys
import numpy as np
from factory import get_problem, get_algorithm
import json
import pickle as p
from glob import glob
import pathlib
from algos.monas.mo_sh import selection

root = pathlib.Path(__file__).parent

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def so_evaluate(search_result: Network, problem):
    evaluation_result = {'Networks': []}
    test_performance, eval_cost = problem.get_test_performance(search_result)
    evaluation_result['Networks'].append({'genotype': search_result.genotype,
                                          'phenotype': f'{problem.search_space.decode(search_result.genotype)}',
                                          'test_acc': test_performance})
    evaluation_result['Evaluation Cost (GPU seconds)'] = eval_cost
    return evaluation_result

def mo_evaluate(search_result: ElitistArchive, problem, **kwargs):
    try:
        size_archive = kwargs['size_archive']
    except KeyError:
        size_archive = -1
    if size_archive != -1:
        ids = selection(search_result.archive, size_archive)
        list_networks = [search_result.archive[i] for i in ids]
    else:
        list_networks = search_result.archive
    evaluation_result = {'Networks': []}
    list_metrics = problem.mo_objective[1:]
    eval_cost = 0.0
    for network_id, network in enumerate(list_networks):
        evaluation_result['Networks'].append(
            {'genotype': network.genotype, 'phenotype': problem.search_space.decode(network.genotype)})
        evaluation_result['Networks'][-1]['test_acc'], eval_time = problem.get_test_performance(network)
        network_scores = network.score[1:]
        for i, metric in enumerate(list_metrics):
            evaluation_result['Networks'][-1][metric] = network_scores[i]
        eval_cost += eval_time
    list_metrics.insert(0, 'err')

    hv = problem.get_hv_value(list_networks, '&'.join(list_metrics))
    evaluation_result['HV'] = round(hv, 6)
    evaluation_result['Evaluation Cost (GPU seconds)'] = int(eval_cost)
    return evaluation_result

def run_evaluate(problem, nas_type, save_path, filename, res_file=None, verbose=True, log=False, **kwargs):
    if res_file is not None:
        search_result, search_cost, total_epoch = p.load(open(res_file, 'rb'))
    else:
        search_result, search_cost, total_epoch = kwargs['search_result'], kwargs['search_cost'], kwargs['total_epoch']
    if nas_type == 'so':
        evaluation_result = so_evaluate(search_result, problem)
    else:
        evaluation_result = mo_evaluate(search_result, problem, **kwargs)
    if verbose:
        print('-> Evaluate:')
        for network_id, info in enumerate(evaluation_result['Networks']):
            print(f'   * Network #{network_id}:')
            for key in info:
                print(f'     + {key}: {info[key]}')
            print()
        for key in list(evaluation_result.keys())[1:]:
            print(f'   * {key}: {evaluation_result[key]}')
        print('-' * 100)
    evaluation_result['Search Cost (GPU seconds)'] = search_cost
    evaluation_result['Search Cost (#Epochs)'] = total_epoch

    if log:
        with open(f'{save_path}/{filename}.json', 'w') as fp:
            json.dump(evaluation_result, fp, indent=4, cls=NumpyEncoder)
    return evaluation_result

def run(problem, nas_type, res_path):
    res_files = glob(res_path + '/results/*.p')

    print('Results Path:', res_path)
    for run_id, _res_path in enumerate(res_files):
        print(f'- RunID: {run_id + 1}\n')
        _ = run_evaluate(_res_path, problem, nas_type, res_path, filename=f'evaluation_results_run{run_id + 1}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' PROBLEM '''
    parser.add_argument('--nas_type', type=str, default='so', help='the problem type', choices=['so', 'mo'])
    parser.add_argument('--res_path', type=str)
    parser.add_argument('--ss', type=str, default='nb201', help='the search space',
                        choices=['nb201', 'nb101', 'nbasr', 'darts'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'],
                        help='dataset for NAS-Bench-201')

    parser.add_argument('--optimizer', type=str, default='MF-NAS', help='the search strategy')
    parser.add_argument('--config_file', type=str, default='./configs/algo_201.yaml', help='the configuration file')

    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    nas_type = args.nas_type
    res_path = args.res_path

    search_space = args.ss
    dataset = args.dataset
    if '201' in search_space:
        search_space += f'_{dataset}'

    opt_name = args.optimizer

    problem, _ = get_problem(search_space)
    config_file = args.config_file
    opt, _ = get_algorithm(opt_name, config_file)

    run(problem, nas_type, res_path)