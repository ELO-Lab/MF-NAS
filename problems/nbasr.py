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

    def zc_evaluate(self, network, **kwargs):
        metric = kwargs['metric']

        h = self.get_h(genotype=network.genotype)
        if metric in ['FLOPs', 'params']:
            score = self.benchmark_database[h][metric]
        else:
            score = self.zc_database[h][metric]
        info = {metric: score}
        return info, 0.0

    def train(self, network, **kwargs):
        iepoch = kwargs['iepoch']

        h = self.get_h(genotype=network.genotype)
        info = self.benchmark_database[h]
        all_info = {
            'val_per': -info['val_per'][iepoch - 1]  # for maximization
        }
        train_time = 0
        network.info['cur_iepoch'].append(iepoch)
        network.info['train_time'].append(train_time)
        return all_info, train_time

    def get_test_performance(self, network, **kwargs):
        h = self.get_h(genotype=network.genotype)
        test_PER = np.round(self.benchmark_database[h]['test_per'] * 100, 2)
        return test_PER, 0.0

    def get_h(self, genotype):
        if isinstance(genotype, list):
            genotype = np.array(genotype)
        if isinstance(genotype, np.ndarray):
            genotype = self.search_space.decode(genotype)
        g, _ = get_model_graph(genotype, ops=None, minimize=True)
        h = graph_hash(g)
        return h
