import yaml
from algos import *
from problems import *
from utils import print_info

def get_problem(name):
    with open('configs/problem.yaml', 'r') as file:
        all_configs = yaml.safe_load(file)
    configs = all_configs[name]
    max_eval, max_time, dataset = configs['max_eval'], configs['max_time'], configs['dataset']
    info_problem = {
        'Search space': name.split("_")[0].upper(),
        'Dataset': dataset.upper(),
        'Maximum budget (seconds)': max_time,
        'Maximum budget (evaluations)': max_eval,
    }
    print('Problem:')
    print_info(info_problem)

    if '201' in name:
        return NB_201(max_eval, max_time, dataset), info_problem
    elif name == 'nb101':
        return NB_101(max_eval, max_time, dataset), info_problem
    elif name == 'nbasr':
        return NB_ASR(max_eval, max_time, dataset), info_problem
    else:
        raise ValueError(f'Not support this problem: {name}.')

def get_algorithm(name, config_file):
    f = open(config_file, 'r')
    all_configs = yaml.safe_load(f)

    configs = all_configs[name]
    if name == 'FLS':
        algo = IteratedLocalSearch(first_improvement=True)
    elif name == 'BLS':
        algo = IteratedLocalSearch(first_improvement=False)
    elif name == 'RS':
        algo = RandomSearch()
    elif 'REA' in name:
        algo = REA()
    elif name == 'SH':
        algo = SuccessiveHalving()
    elif name == 'MF-NAS':
        algo = MF_NAS()
    else:
        raise ValueError(f'Not support this algorithm: {name}')
    print(f'Algorithm: {name}')
    algo.set(configs)
    print_info(configs)
    return algo, configs
