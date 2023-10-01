import argparse
import logging
import sys
import numpy as np
from factory import get_problem, get_algorithm


def run(kwargs):
    search_space = kwargs.ss
    dataset = kwargs.dataset
    if '201' in search_space:
        search_space += f'_{dataset}'
    problem = get_problem(search_space)

    opt_name = kwargs.optimizer
    opt = get_algorithm(opt_name)
    opt.adapt(problem)

    n_run = kwargs.n_run
    verbose = kwargs.verbose
    trend_performance, trend_search_cost, trend_total_epoch = [], [], []
    for run_id in range(1, n_run + 1):
        network, search_cost, total_epoch = opt.run(seed=run_id)
        test_performance = problem.get_test_performance(network)
        if verbose:
            network_phenotype = problem.search_space.decode(network.genotype)
            print('RunID:', run_id)
            print('Best architecture found:', network_phenotype)
            print('Performance:', test_performance)
            print('Search cost (in seconds):', search_cost)
            print('Search cost (in epochs):', total_epoch)
            print('-'*100)
        trend_performance.append(test_performance)
        trend_search_cost.append(search_cost)
        trend_total_epoch.append(total_epoch)
    print('Mean:', np.round(np.mean(trend_performance), 2), '\t Std:', np.round(np.std(trend_performance), 2))
    print('Search cost (in seconds):', np.round(np.mean(trend_search_cost), 2))
    print('Search cost (in epochs):', np.round(np.mean(trend_total_epoch), 2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' PROBLEM '''
    parser.add_argument('--ss', type=str, default='nb201', help='the search space',
    choices=['nb201', 'nb101', 'nbasr'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'],
    help='dataset for NAS-Bench-201')

    ''' ALGORITHM '''
    parser.add_argument('--optimizer', type=str, default='MF-NAS', help='the search strategy',
    choices=['RS', 'SH', 'FLS', 'BLS', 'REA', 'REA+W', 'MF-NAS'])

    ''' ENVIRONMENT '''
    parser.add_argument('--n_run', type=int, default=31, help='number of experiment runs')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    run(args)