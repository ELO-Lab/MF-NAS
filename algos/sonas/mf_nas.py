"""
- Source code for "Efficient Multi-Fidelity Neural Architecture Search with Zero-Cost Proxy-Guided Local Search (MF-NAS)"
- Published in GECCO 2024
- Authors: Quan Minh Phan, Ngoc Hoang Luong
"""

from algos import Algorithm
from algos.sonas import IteratedLocalSearch, RandomSearch, REA, GA, SuccessiveHalving
import numpy as np
from models import Network

class MF_NAS(Algorithm):
    """
        GECCO version
    """
    def __init__(self):
        super().__init__()
        # Stage 1: Training-free Search (Local Search at current)
        self.using_zc_metric_stage1 = True
        self.metric_stage1 = None
        self.max_eval_stage1 = None
        self.optimizer_stage1 = None

        # Stage 2: Training-based Search (Successive Halving)
        self.using_zc_metric_stage2 = False
        self.metric_stage2 = None
        self.n_candidate = -1
        self.list_iepoch = None

        self.list_explored_network = []
        self.xx = None

    def _run(self, **kwargs):
        assert self.max_eval_stage1 is not None
        assert self.metric_stage1 is not None
        assert self.metric_stage2 is not None
        assert self.list_iepoch is not None

        best_network = self.search(**kwargs)
        return best_network, self.total_time, self.total_epoch

    def search(self, **kwargs):
        self.n_candidate = int(self.n_candidate)
        # Stage 1: Training-free Search
        if self.optimizer_stage1 == 'FLS':
            optimizer_stage1 = IteratedLocalSearch(first_improvement=True)
        elif self.optimizer_stage1 == 'BLS':
            optimizer_stage1 = IteratedLocalSearch(first_improvement=False)
        elif self.optimizer_stage1 == 'RS':
            optimizer_stage1 = RandomSearch()
        elif self.optimizer_stage1 == 'REA':
            optimizer_stage1 = REA()
        elif self.optimizer_stage1 == 'GA':
            optimizer_stage1 = GA()
        else:
            raise ValueError(f'Not support this optimizer in MF-NAS framework: {self.optimizer_stage1}')

        if self.optimizer_stage1 == 'REA':
            optimizer_stage1.pop_size = 10
            optimizer_stage1.tournament_size = 10
            optimizer_stage1.prob_mutation = 1.0
        elif self.optimizer_stage1 == 'GA':
            optimizer_stage1.pop_size = 10
            optimizer_stage1.tournament_size = 2
            optimizer_stage1.prob_c = 0.9
            optimizer_stage1.prob_m = 1.0
            optimizer_stage1.crossover_method = '2X'

        optimizer_stage1.adapt(self.problem)
        optimizer_stage1.using_zc_metric = self.using_zc_metric_stage1

        _ = optimizer_stage1.search(max_time=self.problem.max_time, max_eval=self.max_eval_stage1,
                                    metric=self.metric_stage1, **kwargs)
        self.total_time += optimizer_stage1.total_time
        self.total_epoch += optimizer_stage1.total_epoch
        self.xx = optimizer_stage1

        network_history_stage1 = optimizer_stage1.network_history[:self.max_eval_stage1]
        score_history_stage1 = optimizer_stage1.score_history[:self.max_eval_stage1]

        ## Remove duplication
        hashes_history_stage1 = np.array([self.problem.get_hash(network) for network in network_history_stage1])
        _, I = np.unique(hashes_history_stage1, return_index=True)
        network_history_stage1 = np.array(network_history_stage1)[I]
        score_history_stage1 = np.array(score_history_stage1)[I]

        self.list_explored_network.append(hashes_history_stage1)

        ## Sort
        I = np.flip(np.argsort(score_history_stage1))
        network_history_stage1 = network_history_stage1[I]

        # Stage 2: Training-based Selection
        ## Get top-k best solutions in terms of training-free metric value. They are the input of SH.
        topK_found_solutions = network_history_stage1[:self.n_candidate]

        ## Initialize Successive Halving
        optimizer_stage2 = SuccessiveHalving()
        optimizer_stage2.adapt(self.problem)
        optimizer_stage2.using_zc_metric = self.using_zc_metric_stage2
        optimizer_stage2.metric = self.metric_stage2
        optimizer_stage2.list_iepoch = self.list_iepoch

        best_network = optimizer_stage2.search(topK_found_solutions, max_time=self.problem.max_time - optimizer_stage1.total_time)
        self.total_time += optimizer_stage2.total_time
        self.total_epoch += optimizer_stage2.total_epoch

        return best_network

class R_MF_NAS(Algorithm):
    """
        Get top-(k/3) candidates + Sample (randomly) k/3 start and k/3 middle nodes in STN.
    """
    def __init__(self):
        super().__init__()
        # Stage 1: Training-free Search (Local Search at current)
        self.using_zc_metric_stage1 = True
        self.metric_stage1 = None
        self.max_eval_stage1 = None
        self.optimizer_stage1 = None

        # Stage 2: Training-based Search (Successive Halving)
        self.using_zc_metric_stage2 = False
        self.metric_stage2 = None
        self.n_candidate = -1
        self.list_iepoch = None

        self.list_explored_network = []
        self.xx = None

    def _run(self, **kwargs):
        assert self.max_eval_stage1 is not None
        assert self.metric_stage1 is not None
        assert self.metric_stage2 is not None
        assert self.list_iepoch is not None

        best_network = self.search(**kwargs)
        return best_network, self.total_time, self.total_epoch

    def search(self, **kwargs):
        self.n_candidate = int(self.n_candidate)
        # Stage 1: Training-free Search
        if self.optimizer_stage1 == 'FLS':
            optimizer_stage1 = IteratedLocalSearch(first_improvement=True)
        elif self.optimizer_stage1 == 'BLS':
            optimizer_stage1 = IteratedLocalSearch(first_improvement=False)
        elif self.optimizer_stage1 == 'RS':
            optimizer_stage1 = RandomSearch()
        elif self.optimizer_stage1 == 'REA':
            optimizer_stage1 = REA()
        elif self.optimizer_stage1 == 'GA':
            optimizer_stage1 = GA()
        else:
            raise ValueError(f'Not support this optimizer in MF-NAS framework: {self.optimizer_stage1}')

        if self.optimizer_stage1 == 'REA':
            optimizer_stage1.pop_size = 10
            optimizer_stage1.tournament_size = 10
            optimizer_stage1.prob_mutation = 1.0
        elif self.optimizer_stage1 == 'GA':
            optimizer_stage1.pop_size = 10
            optimizer_stage1.tournament_size = 2
            optimizer_stage1.prob_c = 0.9
            optimizer_stage1.prob_m = 1.0
            optimizer_stage1.crossover_method = '2X'

        optimizer_stage1.adapt(self.problem)
        optimizer_stage1.using_zc_metric = self.using_zc_metric_stage1

        _ = optimizer_stage1.search(max_time=self.problem.max_time, max_eval=self.max_eval_stage1,
                                    metric=self.metric_stage1, **kwargs)
        self.total_time += optimizer_stage1.total_time
        self.total_epoch += optimizer_stage1.total_epoch
        self.xx = optimizer_stage1

        network_history_stage1 = optimizer_stage1.network_history[:self.max_eval_stage1]
        score_history_stage1 = optimizer_stage1.score_history[:self.max_eval_stage1]

        ## Remove duplication
        hashes_history_stage1 = np.array([self.problem.get_hash(network) for network in network_history_stage1])
        unique_hashes_history_stage1, I = np.unique(hashes_history_stage1, return_index=True)
        network_history_stage1 = np.array(network_history_stage1)[I]
        score_history_stage1 = np.array(score_history_stage1)[I]

        self.list_explored_network.append(hashes_history_stage1)

        ## Sort
        I = np.flip(np.argsort(score_history_stage1))
        network_history_stage1 = network_history_stage1[I]
        unique_hashes_history_stage1 = unique_hashes_history_stage1[I]

        # Stage 2: Training-based Selection
        n_topk = int(np.ceil(self.n_candidate/3))
        topK_found_solutions = network_history_stage1[:n_topk].tolist()
        topK_found_hashes = unique_hashes_history_stage1[:n_topk].tolist()

        start, middle, end = {'X': [], 'h': [], 'scores': []}, {'X': [], 'h': [], 'scores': []}, {'X': [], 'h': [], 'scores': []}
        for stage in optimizer_stage1.trend_scores:
            for info in stage:
                if info[-1] == 's' and info[1] not in topK_found_hashes:
                    start['X'].append(info[0])
                    start['h'].append(info[1])
                    start['scores'].append(info[2])
                elif info[-1] == 'm' and info[1] not in topK_found_hashes:
                    middle['X'].append(info[0])
                    middle['h'].append(info[1])
                    middle['scores'].append(info[2])
        start['X'] = start['X'][1:]
        start['h'] = start['h'][1:]
        start['scores'] = start['scores'][1:]

        _, I = np.unique(start['h'], return_index=True)
        start['X'] = np.array(start['X'])[I]
        start['h'] = np.array(start['h'])[I]
        start['scores'] = np.array(start['scores'])[I]
        sample_start_stage = min(int(np.floor(self.n_candidate / 3)), len(start['X']))

        _, I = np.unique(middle['h'], return_index=True)
        middle['X'] = np.array(middle['X'])[I]
        middle['h'] = np.array(middle['h'])[I]
        middle['scores'] = np.array(middle['scores'])[I]

        I = []
        for i, h in enumerate(middle['h']):
            if h not in start['h']:
                I.append(i)
        I = np.array(I)
        if len(I) != 0:
            middle['X'] = np.array(middle['X'])[I]
            middle['h'] = np.array(middle['h'])[I]
            middle['scores'] = np.array(middle['scores'])[I]
        n_add = 0
        if len(start['X']) < int(np.ceil(self.n_candidate / 3)):
            n_add = int(np.ceil(self.n_candidate / 3)) - len(start['X'])
        sample_middle_stage = min(self.n_candidate - len(topK_found_hashes) - sample_start_stage + n_add, len(middle['X']))

        list_n_samples = [int(sample_middle_stage), int(sample_start_stage)]

        for stage_i, obj in enumerate([middle, start]):
            n_selection = list_n_samples[stage_i]
            while n_selection != 0:
                pool = getSolution(list_n_samples[stage_i], obj)

                for i, genotype in enumerate(pool['X']):
                    if pool['h'][i] not in topK_found_hashes:
                        n_selection -= 1
                        topK_found_hashes.append(pool['h'][i])
                        network = Network()
                        network.genotype = genotype
                        topK_found_solutions.append(network)
                        if n_selection == 0:
                            break

        topK_found_solutions = np.array(topK_found_solutions)

        # Stage 2: Training-based Selection
        ## Initialize Successive Halving
        optimizer_stage2 = SuccessiveHalving()
        optimizer_stage2.adapt(self.problem)
        optimizer_stage2.using_zc_metric = self.using_zc_metric_stage2
        optimizer_stage2.metric = self.metric_stage2
        optimizer_stage2.list_iepoch = self.list_iepoch

        best_network = optimizer_stage2.search(topK_found_solutions, max_time=self.problem.max_time - optimizer_stage1.total_time)
        self.total_time += optimizer_stage2.total_time
        self.total_epoch += optimizer_stage2.total_epoch

        return best_network


def getSolution(k: int, pool: dict):
    if len(pool['X']) <= k:
        return pool
    # I = np.flip(np.argsort(pool['scores']))[:k]
    I = np.random.choice(range(len(pool['X'])), size=k, replace=False)
    pool['X'] = pool['X'][I]
    pool['h'] = pool['h'][I]
    pool['scores'] = pool['scores'][I]
    return pool


