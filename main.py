import argparse
import logging
import sys
from factory import get_problem, get_algorithm

def run(kwargs):
    search_space = kwargs.ss
    dataset = kwargs.dataset
    if '201' in search_space:
        search_space += f'_{dataset}'
    problem = get_problem(search_space)

    opt_name, metric = kwargs.optimizer, kwargs.metric
    opt = get_algorithm(opt_name)
    opt.adapt(problem, metric)

    n_run = kwargs.n_run
    for run_id in range(1, n_run+1):
        opt.run(seed=run_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' PROBLEM '''
    parser.add_argument('--ss', type=str, default='nb201', help='the search space',
                        choices=['nb201', 'nb101', 'nbasr'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'],
                        help='dataset for NAS-Bench-201')
    parser.add_argument('--max_eval', type=int, default=3000, help='the maximum number of evaluations')

    ''' ALGORITHM '''
    parser.add_argument('--optimizer', type=str, default='MF-NAS', help='the search strategy',
                        choices=['RS', 'SH', 'ILS', 'BLS', 'REA', 'REA+W', 'MF-NAS'])
    parser.add_argument('--metric', type=str, default='val_acc', help='the performance metric')
    ''' ENVIRONMENT '''
    parser.add_argument('--n_run', type=int, default=31, help='number of experiment runs')
    parser.add_argument('--init_seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', type=int, default=0, help='debug mode')
    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    run(args)