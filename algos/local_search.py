from . import Algorithm
from models import Network
from copy import deepcopy
import itertools
import numpy as np
from .utils import sampling_solution, update_log

class IteratedLocalSearch(Algorithm):
    def __init__(self, first_improvement=True):
        super().__init__()
        self.first_improvement = first_improvement

        self.trend_best_network = []
        self.trend_time = []
        self.network_history = []
        self.score_history = []

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

    def search(self, **kwargs):
        max_eval = kwargs['max_eval']
        max_time = kwargs['max_time']
        metric = kwargs['metric']

        # Initialize starting solution
        init_network = sampling_solution(problem=self.problem)

        cost_time = self.evaluate(init_network, using_zc_metric=self.using_zc_metric, metric=metric)
        self.total_time += cost_time
        self.total_epoch += self.iepoch

        cur_network, best_network = deepcopy(init_network), deepcopy(init_network)
        update_log(best_network=best_network, cur_network=cur_network, algorithm=self)

        while (self.n_eval <= max_eval) and (self.total_time <= max_time):
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
                    cost_time = self.evaluate(neighbor_network, using_zc_metric=self.using_zc_metric, metric=metric)
                    self.total_time += cost_time
                    self.total_epoch += self.iepoch

                    ## Update the best solution so far
                    if neighbor_network.score > best_network.score:
                        best_network = deepcopy(neighbor_network)
                    update_log(best_network=best_network, cur_network=neighbor_network, algorithm=self)

                    ## Update the current solution
                    if neighbor_network.score >= cur_network.score:
                        cur_network = deepcopy(neighbor_network)
                        improved = True

                        if self.first_improvement:
                            break
                if self.first_improvement and improved:
                    break

            # If the current solution cannot be improved, the algorithm is stuck.
            # Therefore, we perform the escape operator.
            if not improved:
                ## Get all neighbors within the distance k = 2 (Escape Operator)
                list_ids = get_indices(cur_network.genotype, 2)
                i = np.random.choice(range(len(list_ids)))
                selected_ids = list_ids[i]

                list_neighbors = get_neighbors(cur_network=cur_network, ids=selected_ids, problem=self.problem)
                cur_network = deepcopy(list_neighbors[0])

                cost_time = self.evaluate(cur_network, using_zc_metric=self.using_zc_metric, metric=metric)
                self.total_time += cost_time
                self.total_epoch += self.iepoch

                if cur_network.score > best_network.score:
                    best_network = deepcopy(cur_network)
                update_log(best_network=best_network, cur_network=cur_network, algorithm=self)

        return best_network

def get_indices(genotype, distance):
    return list(itertools.combinations(range(len(genotype)), distance))

def get_neighbors(cur_network, ids, problem):
    list_neighbors = []
    list_available_ops = []
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
            list_neighbors.append(neighbor)
    np.random.shuffle(list_neighbors)
    return list_neighbors