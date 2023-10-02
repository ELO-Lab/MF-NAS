from problems import Problem
from search_spaces import SS_101
import json
import numpy as np
import pickle as p
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')
ROOT_DIR = '/'.join(ROOT_DIR[:-1])

time_dict = {
    'synflow': 1.4356617034300945,
    'params': 0.30215823150349097,
    'grad_norm': 2.035946015246093,
    'grasp': 5.570546795804546,
    'jacob_cov': 2.5207841626097856,
    'snip': 2.028758352457235,
    'fisher': 2.610283957422675,
    'flops': 0.30215823150349097,
}


class NB_101(Problem):
    def __init__(self, max_eval, max_time, dataset, **kwargs):
        super().__init__(SS_101(), max_eval, max_time)
        self.dataset = dataset
        self.zc_database = json.load(open(ROOT_DIR + f'/database/nb101/zc_database.json'))
        self.flops_database = p.load(open(ROOT_DIR + f'/database/nb101/flops_data.p', 'rb'))
        self.benchmark_database = p.load(open(ROOT_DIR + f'/database/nb101/data.p', 'rb'))

    def evaluate(self, network, **kwargs):
        using_zc_metric = bool(kwargs['using_zc_metric'])
        if using_zc_metric:
            metric = kwargs['metric']
            time = self.zc_evaluate(network, metric=metric)
        else:
            metric = '_'.join(kwargs['metric'].split('_')[:-1])
            iepoch = int(kwargs['metric'].split('_')[-1])
            time = self.train(network, metric=metric, iepoch=iepoch)
        return time

    def zc_evaluate(self, network, **kwargs):
        metric = kwargs['metric']
        genotype = network.genotype

        time = time_dict[metric]
        h = self.get_h(network=genotype)
        if metric in ['flops', 'params']:
            if metric == 'flops':
                score = self.flops_database[h]
            else:
                score = self.benchmark_database['108'][h]['n_params']
        else:
            score = self.zc_database[h][metric]
        network.score = score
        return time

    def train(self, network, **kwargs):
        metric = kwargs['metric']
        iepoch = kwargs['iepoch']
        genotype = network.genotype

        h = self.get_h(network=genotype)
        info = self.benchmark_database[f'{iepoch}'][h]
        score = info[metric]
        network.score = score
        time = info['train_time'] - network.info['train_time'][-1]
        network.info['cur_iepoch'].append(iepoch)
        network.info['train_time'].append(time)
        return time

    def get_test_performance(self, network, **kwargs):
        genotype = network.genotype
        h = self.get_h(network=genotype)
        test_acc = np.round(self.benchmark_database['108'][h]['test_acc'] * 100, 2)
        return test_acc

    def get_h(self, network):
        if isinstance(network, np.ndarray):
            network = self.search_space.decode(network)
        modelspec = self.search_space.ModelSpec_(network['matrix'], network['ops'])
        h = modelspec.hash_spec(['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'])
        return h
