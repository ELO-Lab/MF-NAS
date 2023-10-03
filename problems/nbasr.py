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
