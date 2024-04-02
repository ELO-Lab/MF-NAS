from problems import Problem
from search_spaces import SS_ASR
from search_spaces.nbasr.utils import get_model_graph, graph_hash
import numpy as np
import pickle as p
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')
ROOT_DIR = '/'.join(ROOT_DIR[:-1])

class NB_ASR(Problem):
    def __init__(self, max_eval, max_time, dataset, **kwargs):
        super().__init__(SS_ASR(), max_eval, max_time)
        self.dataset = dataset
        self.zc_database = p.load(open(ROOT_DIR + f'/database/nbasr/zc_database.p', 'rb'))
        self.benchmark_database = p.load(open(ROOT_DIR + f'/database/nbasr/data.p', 'rb'))

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

        h = self.get_h(network=genotype)
        if metric in ['FLOPs', 'params']:
            score = self.benchmark_database[h][metric]
        else:
            score = self.zc_database[h][metric]
        network.score = score
        return 0.0

    def train(self, network, **kwargs):
        metric = kwargs['metric']
        iepoch = kwargs['iepoch']
        genotype = network.genotype

        h = self.get_h(network=genotype)
        info = self.benchmark_database[h]
        score = info[metric][iepoch - 1]
        network.score = -score  # all problems are maximization ones
        time = 0
        network.info['cur_iepoch'].append(iepoch)
        network.info['train_time'].append(time)
        return time

    def get_test_performance(self, network, **kwargs):
        genotype = network.genotype
        h = self.get_h(network=genotype)
        test_PER = np.round(self.benchmark_database[h]['test_per'] * 100, 2)
        return test_PER

    def get_h(self, network):
        if isinstance(network, list):
            network = np.array(network)
        if isinstance(network, np.ndarray):
            network = self.search_space.decode(network)
        g, _ = get_model_graph(network, ops=None, minimize=True)
        h = graph_hash(g)
        return h
