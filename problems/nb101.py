from problems import Problem
from search_spaces import SS_101
from search_spaces.nb101.utils import ModelSpec_
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
        self.zc_database = p.load(open(ROOT_DIR + f'/database/nb101/zc_database.p', 'rb'))
        self.flops_database = p.load(open(ROOT_DIR + f'/database/nb101/flops_database.p', 'rb'))
        self.benchmark_database = p.load(open(ROOT_DIR + f'/database/nb101/data.p', 'rb'))

    def zc_evaluate(self, network, **kwargs):
        metric = kwargs['metric']

        time = time_dict[metric]
        h = self.get_h(genotype=network.genotype)
        if metric in ['flops', 'params']:
            if metric == 'flops':
                score = self.flops_database[h]
            else:
                score = self.benchmark_database['108'][h]['n_params']
        else:
            score = self.zc_database[h][metric]
        info = {metric: score}
        return info, time

    def train(self, network, **kwargs):
        iepoch = kwargs['iepoch']

        h = self.get_h(genotype=network.genotype)
        info = self.benchmark_database[f'{iepoch}'][h]
        all_infos = {
            'train_acc': info['train_acc'],
            'val_acc': info['val_acc'],
        }
        train_time = info['train_time'] - network.info['train_time'][-1]
        network.info['cur_iepoch'].append(iepoch)
        network.info['train_time'].append(train_time)
        return all_infos, train_time

    def get_test_performance(self, network, **kwargs):
        h = self.get_h(genotype=network.genotype)
        test_acc = np.round(self.benchmark_database['108'][h]['test_acc'] * 100, 2)
        train_time = self.benchmark_database['108'][h]['train_time']
        return test_acc, train_time

    def get_h(self, genotype):
        if isinstance(genotype, list):
            genotype = np.array(genotype)
        phenotype = self.search_space.decode(genotype)
        modelspec = ModelSpec_(phenotype['matrix'], phenotype['ops'])
        h = modelspec.hash_spec(['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'])
        return h
