from algos import Algorithm
from models import Network
import itertools
import numpy as np
from algos.utils import sampling_solution, compare_f1_f2, ElitistArchive


class ParetoLocalSearch(Algorithm):
    def __init__(self):
        super().__init__(nas_type='mo')
        self.trend_best_network = []
        self.trend_time = []
        self.network_history = []
        self.archive = ElitistArchive()

        self.list_metrics, self.list_iepochs, self.need_trained = [], [], []
        self.visited = []
        self.explored_networks = []

    def _reset(self):
        self.trend_best_network = []
        self.trend_time = []
        self.network_history = []
        self.archive = ElitistArchive()
        self.visited = []
        self.explored_networks = []

    def reformat_list_metrics(self):
        list_metrics = []
        for i in range(len(self.need_trained)):
            if self.need_trained[i]:
                list_metrics.append(self.list_metrics[i] + f'_{self.list_iepochs[i]}')
            else:
                list_metrics.append(self.list_metrics[i])
        return list_metrics

    def finalize(self, **kwargs):
        try:
            save_path = kwargs['save_path']
            rid = kwargs['rid']
            import pickle as p
            p.dump(self.explored_networks, open(save_path + f'/explored_networks_run{rid}.p', 'wb'))
        except KeyError:
            pass

    def _run(self, **kwargs):
        self._reset()
        max_eval = self.problem.max_eval if self.max_eval is None else self.max_eval
        max_time = self.problem.max_time if self.max_time is None else self.max_time
        approximation_set = self.search(max_eval=max_eval, max_time=max_time, **kwargs)
        return approximation_set, self.total_time, self.total_epoch

    def search(self, **kwargs):
        max_eval = kwargs['max_eval']
        max_time = kwargs['max_time']
        list_metrics = self.reformat_list_metrics()

        while (self.n_eval <= max_eval) and (self.total_time <= max_time):
            while True:
                init_network = sampling_solution(self.problem)
                _hash = self.problem.get_hash(init_network)
                if _hash not in self.visited:
                    self.visited.append(_hash)
                    break
            train_time, train_epoch, is_terminated = self.problem.mo_evaluate(list_networks=[init_network],
                                                                              list_metrics=list_metrics,
                                                                              need_trained=self.need_trained,
                                                                              cur_total_time=self.total_time,
                                                                              max_time=max_time)
            self.network_history.append(init_network)
            self.archive.update(init_network, problem=self.problem)
            self.n_eval += 1
            self.total_time += train_time
            self.total_epoch += train_epoch
            self.explored_networks.append([self.total_time, init_network.genotype, self.problem.get_hash(init_network), init_network.score])
            if is_terminated or self.n_eval >= max_eval:
                return self.archive

            F = [init_network]
            X_PLO_NF = [init_network]

            while len(F) != 0:
                i = np.random.choice(range(len(F)))
                x = F[i]
                W = []

                list_ids = get_indices(x.genotype, 1)
                N = []
                for j in list_ids:
                    _neighbors = get_neighbors(cur_network=x, ids=j, problem=self.problem, visited=self.visited)
                    for neighbor in _neighbors:
                        _hash = self.problem.get_hash(neighbor)
                        self.visited.append(_hash)
                    N += _neighbors

                for neighbor in N:
                    train_time, train_epoch, is_terminated = self.problem.mo_evaluate(list_networks=[neighbor],
                                                                                      list_metrics=list_metrics,
                                                                                      need_trained=self.need_trained,
                                                                                      cur_total_time=self.total_time,
                                                                                      max_time=max_time)
                    self.network_history.append(neighbor)
                    self.archive.update(neighbor, problem=self.problem)
                    self.n_eval += 1
                    self.total_time += train_time
                    self.total_epoch += train_epoch
                    self.explored_networks.append([self.total_time, neighbor.genotype, self.problem.get_hash(neighbor), neighbor.score])

                    if is_terminated or self.n_eval >= max_eval:
                        return self.archive

                    if (not is_dominated(neighbor, W)) and (not is_dominated(neighbor, F)) and (not is_dominated(neighbor, X_PLO_NF)):
                        if check_Valid(neighbor, W, self.problem) and check_Valid(neighbor, F, self.problem) and check_Valid(neighbor, X_PLO_NF, self.problem):
                            W.append(neighbor)

                X_PLO_NF = X_PLO_NF + W
                # X_PLO_NF = get_fronts(X_PLO_NF, NF)

                F.remove(F[i])
                F = F + W
                # F = get_fronts(F, NF)
        return self.archive


def get_indices(genotype, distance):
    return list(itertools.combinations(range(len(genotype)), distance))


def get_neighbors(cur_network, ids, problem, visited):
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

    visited_all = visited.copy()
    visited_neighbors = []
    for ops in list_ops:
        genotype_neighbor = np.array(genotype_cur_state).copy()
        genotype_neighbor[ids] = ops
        if problem.search_space.is_valid(genotype_neighbor):
            neighbor = Network()
            neighbor.genotype = genotype_neighbor.tolist().copy()
            _hash = problem.get_hash(neighbor)
            if (_hash not in visited_all) and (_hash not in visited_neighbors):
                list_neighbors.append(neighbor)
                visited_neighbors.append(_hash)
    np.random.shuffle(list_neighbors)
    return list_neighbors


def get_front_0(X):
    l = len(X)
    r = np.zeros(l, dtype=int)
    for i in range(l):
        if r[i] == 0:
            for j in range(i + 1, l):
                better_sol = compare_f1_f2(X[i].score, X[j].score)
                if better_sol == 0:
                    r[j] += 1
                elif better_sol == 1:
                    r[i] += 1
                    break
    return np.array(X)[r == 0].tolist()


def is_dominated(x, Y):
    for y in Y:
        if compare_f1_f2(x.score, y.score) == 1:
            return True
    return False


def check_Valid(x, X, problem):
    hashKey_list = [problem.get_hash(network) for network in X]
    hashKey_x = problem.get_hash(x)
    if hashKey_x not in hashKey_list:
        return True