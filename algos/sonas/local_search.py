from algos import Algorithm
from models import Network
from copy import deepcopy
import itertools
import numpy as np
from algos.utils import sampling_solution, update_log
from tqdm import tqdm

class IteratedLocalSearch(Algorithm):
    def __init__(self, first_improvement=True):
        super().__init__()
        self.first_improvement = first_improvement

        self.trend_best_network = []
        self.trend_time = []
        self.network_history = []
        self.score_history = []
        self.trend_scores = []

    def _reset(self):
        self.trend_best_network = []
        self.trend_time = []
        self.network_history = []
        self.score_history = []

    def _run(self, **kwargs):
        self._reset()
        max_eval = self.problem.max_eval if self.max_eval is None else self.max_eval
        max_time = self.problem.max_time if self.max_time is None else self.max_time
        metric = self.metric + f'_{self.iepoch}' if not self.using_zc_metric else self.metric

        best_network = self.search(max_eval=max_eval, max_time=max_time, metric=metric, **kwargs)
        return best_network, self.total_time, self.total_epoch

    def evaluate(self, network, metric=None, using_zc_metric=None):
        if using_zc_metric is None:
            using_zc_metric = self.using_zc_metric
        if metric is None:
            metric = self.search_metric
        total_time, total_epoch, is_terminated = self.problem.evaluate(network,
                                                                       metric=metric, using_zc_metric=using_zc_metric,
                                                                       cur_total_time=self.total_time,
                                                                       max_time=self.max_time)
        self.n_eval += 1
        self.total_time += total_time
        self.total_epoch += total_epoch
        return is_terminated or self.n_eval >= self.max_eval

    def search(self, **kwargs):
        self.max_eval = kwargs['max_eval']
        self.max_time = kwargs['max_time']
        self.search_metric = kwargs['metric']
        self.trend_scores = []

        # Initialize starting solution
        init_network = sampling_solution(problem=self.problem)

        is_terminated = self.evaluate(init_network)

        cur_network, best_network = deepcopy(init_network), deepcopy(init_network)
        update_log(best_network=best_network, cur_network=cur_network, algorithm=self)

        if is_terminated:
            return best_network

        ID = 0
        self.trend_scores = [[[cur_network.genotype.copy(), self.problem.get_hash(cur_network), cur_network.score, self.problem.get_test_performance(cur_network)[0], 's']]]

        while True:
            improved = False

            # Get all neighbors within the distance k = 1
            list_ids = get_indices(cur_network.genotype, 1)
            while len(list_ids) != 0:
                i = np.random.choice(range(len(list_ids)))
                selected_ids = list_ids[i]
                list_ids.remove(list_ids[i])

                list_neighbors = get_neighbors(cur_network=cur_network, ids=selected_ids, problem=self.problem)
                ## For each neighbor, evaluate and compare to the current solution
                for neighbor_network in list_neighbors:
                    is_terminated = self.evaluate(neighbor_network)

                    ## Update the best solution so far
                    if neighbor_network.score > best_network.score:
                        best_network = deepcopy(neighbor_network)
                    update_log(best_network=best_network, cur_network=neighbor_network, algorithm=self)

                    ## Update the current solution
                    # if neighbor_network.score >= cur_network.score:
                    if neighbor_network.score > cur_network.score:
                        cur_network = deepcopy(neighbor_network)
                        self.trend_scores[ID].append([cur_network.genotype.copy(), self.problem.get_hash(cur_network), cur_network.score, self.problem.get_test_performance(cur_network)[0], 'm'])
                        improved = True

                        if self.first_improvement:
                            break

                    if is_terminated:
                        return best_network

                if self.first_improvement and improved:
                    break

            # If the current solution cannot be improved, the algorithm is stuck.
            # Therefore, we perform the escape operator.
            if not improved:
                if len(self.trend_scores[ID]) > 1:
                    self.trend_scores[ID][-1][-1] = 'e'

                ## Get all neighbors within the distance k = 2 (Escape Operator)
                while True:
                    list_ids = get_indices(cur_network.genotype, 2)
                    i = np.random.choice(range(len(list_ids)))
                    selected_ids = list_ids[i]

                    list_neighbors = get_neighbors(cur_network=cur_network, ids=selected_ids, problem=self.problem)
                    if len(list_neighbors) != 0:
                        break
                cur_network = deepcopy(list_neighbors[0])

                is_terminated = self.evaluate(cur_network)
                ID += 1
                self.trend_scores.append([[cur_network.genotype.copy(), self.problem.get_hash(cur_network), cur_network.score, self.problem.get_test_performance(cur_network)[0], 's']])

                if cur_network.score > best_network.score:
                    best_network = deepcopy(cur_network)
                update_log(best_network=best_network, cur_network=cur_network, algorithm=self)

                if is_terminated:
                    return best_network

def get_indices(genotype, distance):
    return list(itertools.combinations(range(len(genotype)), distance))

def get_neighbors(cur_network, ids, problem):
    list_neighbors = []
    list_available_ops = []
    list_neighbor_hashes = []
    genotype_cur_state = cur_network.genotype.copy()
    for i in ids:
        # Get all neighbors at the index-i (i in list of indices ids)
        # In case of distance == 1, ids only has 1 index
        _available_ops = problem.search_space.return_available_ops(i).copy()
        _available_ops.remove(genotype_cur_state[i])
        list_available_ops.append(_available_ops)
    list_ops = list(itertools.product(*list_available_ops))
    ids = np.array(ids)
    for ops in list_ops:
        genotype_neighbor = np.array(genotype_cur_state).copy()
        genotype_neighbor[ids] = ops
        if problem.search_space.is_valid(genotype_neighbor):
            neighbor = Network()
            neighbor.genotype = genotype_neighbor.tolist().copy()
            _hash = problem.get_hash(neighbor)
            if _hash not in list_neighbor_hashes:
                list_neighbors.append(neighbor)
                list_neighbor_hashes.append(_hash)
    np.random.shuffle(list_neighbors)
    return list_neighbors
