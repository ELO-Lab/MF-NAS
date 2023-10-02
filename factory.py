import yaml
from algos import *
from problems import *

def get_problem(name):
    with open('configs/problem.yaml', 'r') as file:
        all_configs = yaml.safe_load(file)
    configs = all_configs[name]
    max_eval, max_time, dataset = configs['max_eval'], configs['max_time'], configs['dataset']

    if '201' in name:
        return NB_201(max_eval, max_time, dataset)
    elif name == 'nb101':
        return NB_101(max_eval, max_time, dataset)
    elif name == 'nbasr':
        return NB_ASR(max_eval, max_time, dataset)
    else:
        raise ValueError(f'Not support this problem: {name}.')

def get_algorithm(name):
    with open('configs/algo.yaml', 'r') as file:
        all_configs = yaml.safe_load(file)
    configs = all_configs[name]
    if name == 'FLS':
        algo = FirstImprovementLS()
    elif name == 'BLS':
        algo = BestImprovementLS()
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
    algo.set(configs)
    return algo
