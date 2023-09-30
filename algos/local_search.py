from . import Algorithm
from models import Network
from copy import deepcopy
import itertools
import numpy as np

def get_indices(genotype, distance):
    return list(itertools.combinations(range(len(genotype)), distance))

def get_all_neighbors(cur_network, ids, problem):
    list_neighbors = []
    list_available_ops = []
    genotype_cur_state = cur_network.genotype.copy()
    for i in ids:
        # Get all neighbors at the index-i (i in list of indices ids)
        # In case of distance == 1, ids only has 1 index
        _available_ops = problem.search_space.available_ops.copy()
        _available_ops.remove(genotype_cur_state[i])
        list_available_ops.append(_available_ops)
    list_ops = list(itertools.product(*list_available_ops))
    ids = np.array(ids)
    for ops in list_ops:
        genotype_neighbor = np.array(genotype_cur_state).copy()
        genotype_neighbor[ids] = ops
        neighbor = Network()
        neighbor.genotype = genotype_neighbor.tolist()
        list_neighbors.append(neighbor)
    np.random.shuffle(list_neighbors)
    return list_neighbors

class FirstImprovementLS(Algorithm):
    def __init__(self):
        super().__init__()
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
        if not self.using_zc_metric:
            self.metric = self.metric + f'_{self.iepoch}'
        n_eval = 0
        total_time = 0

        init_network = Network()
        init_network.genotype = self.problem.search_space.sample(genotype=True)
        time = self.problem.evaluate(init_network, using_zc_metric=self.using_zc_metric, metric=self.metric)

        n_eval += 1
        total_time += time

        cur_network = deepcopy(init_network)
        best_network = deepcopy(init_network)

        self.trend_best_network = [best_network]
        self.trend_time = [total_time]

        self.network_history, self.score_history = [cur_network], [cur_network.score]
        while (n_eval <= self.problem.max_eval) and (total_time <= self.problem.max_time):
            improved = False
            list_ids = get_indices(cur_network.genotype, 1)
            while len(list_ids) != 0:
                idx = np.random.choice(range(len(list_ids)))
                ids = list_ids[idx]
                list_ids.remove(list_ids[idx])

                list_neighbors = get_all_neighbors(cur_network=cur_network, ids=ids, problem=self.problem)
                for neighbor_network in list_neighbors:
                    time = self.problem.evaluate(neighbor_network, using_zc_metric=self.using_zc_metric, metric=self.metric)
                    self.network_history.append(neighbor_network)
                    self.score_history.append(neighbor_network.score)

                    n_eval += 1
                    total_time += time
                    self.trend_time.append(total_time)

                    # Update the current solution
                    if neighbor_network.score >= cur_network.score:
                        cur_network = deepcopy(neighbor_network)

                        # Update the best solution so far
                        if neighbor_network.score > best_network.score:
                            best_network = deepcopy(neighbor_network)
                        self.trend_best_network.append(best_network)

                        improved = True
                        break
                    else:
                        self.trend_best_network.append(best_network)
                if improved:
                    break

            # If the current solution cannot be improved, the algorithm is stuck
            # Therefore, we perform the escape operator.
            if not improved:
                list_ids = get_indices(cur_network.genotype, 2)
                idx = np.random.choice(range(len(list_ids)))
                ids = list_ids[idx]

                list_neighbors = get_all_neighbors(cur_network=cur_network, ids=ids, problem=self.problem)
                cur_network = list_neighbors[np.random.choice(len(list_neighbors))]

                time = self.problem.evaluate(cur_network, using_zc_metric=self.using_zc_metric, metric=self.metric)
                self.network_history.append(cur_network)
                self.score_history.append(cur_network.score)

                n_eval += 1
                total_time += time

                if cur_network.score > best_network.score:
                    best_network = deepcopy(cur_network)
                self.trend_best_network.append(best_network)
                self.trend_time.append(total_time)
        best_network = self.trend_best_network[-1]
        search_time = total_time
        return best_network, search_time


class BestImprovementLS(Algorithm):
    def __init__(self):
        super().__init__()
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
        if not self.using_zc_metric:
            self.metric = self.metric + f'_{self.iepoch}'
        n_eval = 0
        total_time = 0

        init_network = Network()
        init_network.genotype = self.problem.search_space.sample(genotype=True)
        time = self.problem.evaluate(init_network, using_zc_metric=self.using_zc_metric, metric=self.metric)

        n_eval += 1
        total_time += time

        cur_network = deepcopy(init_network)
        best_network = deepcopy(init_network)

        self.trend_best_network = [best_network]
        self.trend_time = [total_time]

        self.network_history, self.score_history = [cur_network], [cur_network.score]
        while (n_eval <= self.problem.max_eval) and (total_time <= self.problem.max_time):
            improved = False
            list_ids = get_indices(cur_network.genotype, 1)

            all_neighbors = []
            while len(list_ids) != 0:
                idx = np.random.choice(range(len(list_ids)))
                ids = list_ids[idx]
                list_ids.remove(list_ids[idx])

                list_neighbors = get_all_neighbors(cur_network=cur_network, ids=ids, problem=self.problem)
                all_neighbors += list_neighbors

            for neighbor_network in all_neighbors:
                time = self.problem.evaluate(neighbor_network, using_zc_metric=self.using_zc_metric, metric=self.metric)
                self.network_history.append(neighbor_network)
                self.score_history.append(neighbor_network.score)

                n_eval += 1
                total_time += time
                self.trend_time.append(total_time)

                # Update the current solution
                if neighbor_network.score >= cur_network.score:
                    cur_network = deepcopy(neighbor_network)

                    # Update the best solution so far
                    if neighbor_network.score > best_network.score:
                        best_network = deepcopy(neighbor_network)
                    self.trend_best_network.append(best_network)

                    improved = True
                else:
                    self.trend_best_network.append(best_network)

            # If the current solution cannot be improved, the algorithm is stuck
            # Therefore, we perform the escape operator.
            if not improved:
                list_ids = get_indices(cur_network.genotype, 2)
                idx = np.random.choice(range(len(list_ids)))
                ids = list_ids[idx]

                list_neighbors = get_all_neighbors(cur_network=cur_network, ids=ids, problem=self.problem)
                cur_network = list_neighbors[np.random.choice(len(list_neighbors))]

                time = self.problem.evaluate(cur_network, using_zc_metric=self.using_zc_metric, metric=self.metric)
                self.network_history.append(cur_network)
                self.score_history.append(cur_network.score)

                n_eval += 1
                total_time += time

                if cur_network.score > best_network.score:
                    best_network = deepcopy(cur_network)
                self.trend_best_network.append(best_network)
                self.trend_time.append(total_time)
        best_network = self.trend_best_network[-1]
        search_time = total_time
        return best_network, search_time
