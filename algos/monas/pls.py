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
        self.score_history = []
        self.archive = ElitistArchive()

        self.list_metrics, self.list_iepochs, self.need_trained = [], [], []

    def _reset(self):
        self.trend_best_network = []
        self.trend_time = []
        self.network_history = []
        self.score_history = []
        self.archive = ElitistArchive()

    def _run(self, **kwargs):
        self._reset()
        max_eval = self.problem.max_eval if self.max_eval is None else self.max_eval
        max_time = self.problem.max_time if self.max_time is None else self.max_time
        list_metrics = []
        for i in range(len(self.need_trained)):
            if self.need_trained[i]:
                list_metrics.append(self.list_metrics[i] + f'_{self.list_iepochs[i]}')
            else:
                list_metrics.append(self.list_metrics[i])
        approximation_set = self.search(max_eval=max_eval, max_time=max_time, list_metrics=list_metrics, **kwargs)
        return approximation_set, self.total_time, self.total_epoch

    def search(self, **kwargs):
        max_eval = kwargs['max_eval']
        max_time = kwargs['max_time']
        list_metrics = kwargs['list_metrics']

        while (self.n_eval <= max_eval) and (self.total_time <= max_time):
            init_network = sampling_solution(self.problem)
            train_time, train_epoch, is_terminated = self.problem.mo_evaluate(list_networks=[init_network],
                                                                              list_metrics=list_metrics,
                                                                              need_trained=self.need_trained,
                                                                              cur_total_time=self.total_time,
                                                                              max_time=max_time)
            self.network_history.append(init_network)
            self.archive.update(init_network)
            self.n_eval += 1
            self.total_time += train_time
            self.total_epoch += train_epoch
            # init_network()

            if is_terminated:
                return self.archive

            F = [init_network]
            X_PLO_NF = [init_network]

            while len(F) != 0:
                i = 0
                x = F[i]
                W = []

                list_ids = get_indices(x.genotype, 1)
                N = []
                for j in list_ids:
                    _neighbors = get_neighbors(cur_network=x, ids=j, problem=self.problem)
                    N += _neighbors

                for neighbor in N:
                    train_time, train_epoch, is_terminated = self.problem.mo_evaluate(list_networks=[neighbor],
                                                                                      list_metrics=list_metrics,
                                                                                      need_trained=self.need_trained,
                                                                                      cur_total_time=self.total_time,
                                                                                      max_time=max_time)
                    self.network_history.append(neighbor)
                    self.archive.update(neighbor)
                    self.n_eval += 1
                    self.total_time += train_time
                    self.total_epoch += train_epoch
                    # neighbor()

                    if is_terminated:
                        return self.archive

                    if (not is_dominated(neighbor, W)) and (not is_dominated(neighbor, F)) and (not is_dominated(neighbor, X_PLO_NF)):
                        if check_Valid(neighbor, W) and check_Valid(neighbor, F) and check_Valid(neighbor, X_PLO_NF):
                            W.append(neighbor)

                X_PLO_NF = X_PLO_NF + W
                # X_PLO_NF = get_fronts(X_PLO_NF, NF)

                F.remove(F[i])
                F = F + W
                # F = get_fronts(F, NF)
        return self.archive

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

def check_Valid(x, X):
    hashKey_list = [''.join(map(str, x_.genotype)) for x_ in X]
    hashKey_x = ''.join(map(str, x.genotype))
    if hashKey_x not in hashKey_list:
        return True