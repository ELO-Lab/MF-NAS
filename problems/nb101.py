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

    def evaluate(self, networks, **kwargs):
        max_time = kwargs['max_time']
        cur_total_time = kwargs['cur_total_time']
        TOTAL_TIME, TOTAL_EPOCHS = 0.0, 0
        if not isinstance(networks, list) and not isinstance(networks, np.ndarray):
            networks = [networks]
        for _network in networks:
            cur_score = _network.score
            train_time, train_epoch = self._evaluate(_network, bool(kwargs['using_zc_metric']), kwargs['metric'])
            if cur_total_time + TOTAL_TIME + train_time > max_time:
                _network.score = cur_score
                return TOTAL_TIME, TOTAL_EPOCHS, True
            TOTAL_TIME += train_time
            TOTAL_EPOCHS += train_epoch
        return TOTAL_TIME, TOTAL_EPOCHS, False

    def _evaluate(self, network, using_zc_metric, metric):
        total_epoch = 0
        if using_zc_metric:
            _metric = metric
            total_time = self.zc_evaluate(network, metric=_metric)
        else:
            _metric = '_'.join(metric.split('_')[:-1])
            iepoch = int(metric.split('_')[-1])
            total_time, total_epoch = self.train(network, metric=_metric, iepoch=iepoch)
        return total_time, total_epoch

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
        if isinstance(network, list):
            network = np.array(network)
        if isinstance(network, np.ndarray):
            network = self.search_space.decode(network)
        modelspec = ModelSpec_(network['matrix'], network['ops'])
        h = modelspec.hash_spec(['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'])
        return h
