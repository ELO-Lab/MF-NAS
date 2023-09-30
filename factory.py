import yaml
from problems import NB_201
from algos import (
    FirstImprovementLS
)

def get_problem(name):
    with open('configs/problem.yaml', 'r') as file:
        all_configs = yaml.safe_load(file)
    configs = all_configs[name]
    max_eval, max_time, dataset = configs['max_eval'], configs['max_time'], configs['dataset']
    if '201' in name:
        return NB_201(max_eval, max_time, dataset)
    elif name == 'nb101':
        pass
    elif name == 'nbasr':
        pass
    else:
        raise ValueError(f'Not support this problem: {name}.')

def get_algorithm(name):
    if name == 'FLS':
        return FirstImprovementLS()
    else:
        raise ValueError(f'Not support this algorithm: {name}')