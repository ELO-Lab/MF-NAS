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
        if not self.using_zc_metric and self.iepoch is None:
            raise ValueError

        best_network = self.search(max_eval=max_eval, max_time=max_time, metric=self.metric, iepoch=self.iepoch, **kwargs)
        return best_network, self.total_time, self.total_epoch

    def search(self, **kwargs):  # ils
        max_eval, max_time = kwargs['max_eval'], kwargs['max_time']
        metric, iepoch = kwargs['metric'], kwargs['iepoch']

        # Initialize starting solution
        init_sol = sampling_solution(problem=self.problem)
        info, cost_time = self.evaluate(init_sol, using_zc_metric=self.using_zc_metric, metric=metric, iepoch=iepoch)
        init_sol.score = info[metric]

        self.total_time += cost_time
        self.total_epoch += self.iepoch

        update_log(best_network=init_sol, cur_network=init_sol, algorithm=self)

        lo = self.local_search(init_sol, metric, iepoch)
        best_lo = deepcopy(lo)

        while (self.n_eval < max_eval) and (self.total_time < max_time):
            s = deepcopy(best_lo)
            s = self.escape_operator(s)
            info, cost_time = self.evaluate(s, using_zc_metric=self.using_zc_metric, metric=metric, iepoch=iepoch)
            s.score = info[metric]

            self.total_time += cost_time
            self.total_epoch += self.iepoch
            if s.score > self.trend_best_network[-1].score:
                update_log(best_network=s, cur_network=s, algorithm=self)
            else:
                update_log(best_network=self.trend_best_network[-1], cur_network=s, algorithm=self)

            lo = self.local_search(s, metric, iepoch)

            if lo.score > best_lo.score:
                best_lo = deepcopy(lo)

        return best_lo

    def local_search(self, init_sol, metric, iepoch):
        improved, sol = self.neighbor_explorer(init_sol, metric, iepoch)
        while improved:
            improved, sol = self.neighbor_explorer(sol, metric, iepoch)
        return sol

    def neighbor_explorer(self, sol, metric, iepoch):
        improved = False
        best_sol = deepcopy(sol)

        # Get all neighbors within the distance k = 1
        list_ids = get_indices(sol.genotype, 1)
        all_neighbors = []
        while len(list_ids) != 0:
            i = np.random.choice(range(len(list_ids)))
            selected_ids = list_ids[i]
            list_ids.remove(list_ids[i])

            list_neighbors = get_neighbors(cur_network=sol, ids=selected_ids, problem=self.problem)
            all_neighbors += list_neighbors
        np.random.shuffle(all_neighbors)

        # For each neighbor, evaluate and compare to the current solution
        for new_sol in all_neighbors:
            info, cost_time = self.evaluate(new_sol, using_zc_metric=self.using_zc_metric, metric=metric, iepoch=iepoch)
            new_sol.score = info[metric]

            self.total_time += cost_time
            self.total_epoch += self.iepoch

            if new_sol.score > self.trend_best_network[-1].score:
                update_log(best_network=new_sol, cur_network=new_sol, algorithm=self)
            else:
                update_log(best_network=self.trend_best_network[-1], cur_network=new_sol, algorithm=self)

            if new_sol.score > sol.score:
                improved = True
                if new_sol.score > best_sol.score:
                    best_sol = deepcopy(new_sol)
                if self.first_improvement:
                    break
        if improved:
            return improved, best_sol
        else:
            return improved, sol

    def escape_operator(self, sol):
        # Get all neighbors within the distance k = 2
        list_ids = get_indices(sol.genotype, 2)
        all_neighbors = []
        while len(list_ids) != 0:
            i = np.random.choice(range(len(list_ids)))
            selected_ids = list_ids[i]
            list_ids.remove(list_ids[i])

            list_neighbors = get_neighbors(cur_network=sol, ids=selected_ids, problem=self.problem)
            all_neighbors += list_neighbors
        np.random.shuffle(all_neighbors)

        # Choose a random neighbor
        new_sol = deepcopy(all_neighbors[0])
        return new_sol

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