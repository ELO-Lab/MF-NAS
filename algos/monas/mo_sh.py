import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler

from algos import Algorithm
from algos.utils import sampling_solution, ElitistArchive


class MultiObjective_SuccessiveHalving(Algorithm):
    def __init__(self):
        super().__init__()
        self.list_metrics = None
        self.list_iepochs = None
        self.need_trained = None
        self.n_remaining_candidates = None
        self.archive = ElitistArchive()

    def _run(self):
        assert self.list_metrics is not None
        assert len(self.list_iepochs[0]) is not None

        max_time = self.problem.max_time if self.max_time is None else self.max_time

        list_network = self.sample()
        approximation_set = self.search(list_network, max_time=max_time)
        return approximation_set, self.total_time, self.total_epoch

    def sample(self):
        list_networks = []
        hashes = []
        while True:
            network = sampling_solution(self.problem)
            _hash = self.problem.get_hash(network)
            if _hash not in hashes:
                list_networks.append(network)
                hashes.append(_hash)
            if len(hashes) == self.n_remaining_candidates[0]:
                return list_networks

    def search(self, list_network, **kwargs):
        assert len(list_network) != 0
        max_time = kwargs['max_time']

        list_network = np.array(list_network)

        list_metrics = []
        for iepoch in self.list_iepochs[0]:
            _metric = self.list_metrics[0] + f'_{iepoch}'
            metrics = self.list_metrics[1:].copy()
            metrics.insert(0, _metric)
            list_metrics.append(metrics)

        n_remaining_candidates = self.n_remaining_candidates[1:]
        for i, _list_metrics in enumerate(list_metrics):
            evaluated_network = []

            train_time, train_epoch, is_terminated = self.problem.mo_evaluate(list_network,
                                                                              list_metrics=_list_metrics,
                                                                              need_trained=self.need_trained,
                                                                              cur_total_time=self.total_time,
                                                                              max_time=max_time)
            self.total_time += train_time
            self.total_epoch += train_epoch

            for network in list_network:
                evaluated_network.append(network)

            iepoch = int(_list_metrics[0].split('_')[-1])
            if is_terminated or iepoch == self.list_iepochs[0][-1]:
                for network in list_network:
                    self.archive.update_without_check(network)
                return self.archive

            ids = selection(list_network, n_remaining_candidates[i])
            list_network = np.array(evaluated_network)[ids]


def selection(list_network, n_survive):
    F = np.array([network.score for network in list_network])
    fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)
    total_idv = sum([len(front) for front in fronts])
    if total_idv <= n_survive:
        ids = [i for front in fronts for i in front]
        return ids
    F = np.array([list_network[i].score for i in fronts[-1]])
    _ids = greedy_selection(F, n_survive - (total_idv - len(fronts[-1])))
    _fronts = np.array([fronts[-1][i] for i in _ids])
    fronts[-1] = _fronts.copy()
    ids = [i for front in fronts for i in front]
    return ids


def greedy_selection(front, n_survive, **kwargs):
    F = front.copy()
    F = np.abs(F)
    scaler = MinMaxScaler()
    F = scaler.fit_transform(F)
    front = [{'fitness': [f], 'index': i} for i, f in enumerate(F)]
    selected_solutions = []

    front = list(front)
    front = sorted(front, key=lambda x: -x['fitness'][-1][0])

    cur_selected_indices = set([])

    selected_solutions.append(front[0])
    cur_selected_indices.add(0)
    while len(selected_solutions) < n_survive:
        points1 = np.array([x['fitness'][-1] for x in front])
        points2 = np.array([x['fitness'][-1] for x in selected_solutions])

        distances = euclidean_distances(points1, points2)
        cur_min_distances = np.min(distances, axis=1)

        ind_with_max_dist = -1
        max_dist = -float("inf")
        for j in range(len(front)):
            if j not in cur_selected_indices and cur_min_distances[j] > max_dist:
                max_dist = cur_min_distances[j]
                ind_with_max_dist = j
        selected_solutions.append(front[ind_with_max_dist])
        cur_selected_indices.add(ind_with_max_dist)
    I = np.unique([s['index'] for s in selected_solutions])
    return I
