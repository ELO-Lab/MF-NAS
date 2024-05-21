"""
Source code for Local-search algorithm for Multi-Objective Neural Architecture Search (LOMONAS)
GECCO 2023
Authors: Quan Minh Phan, Ngoc Hoang Luong
"""
import numpy as np
from copy import deepcopy
import itertools
from algos.utils import sampling_solution, ElitistArchive, Footprint, not_existed

from algos.base import Algorithm
from models import Network
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.misc import find_duplicates

sorter = NonDominatedSorting()
#################################################### LOMONAS #######################################################
class LOMONAS(Algorithm):
    def __init__(self, name='LOMONAS',
                 archive=None, footprint=None, res_logged=None, **kwargs):
        """
        - name (str) -> the algorithm name (i.e., LOMONAS)
        - k (int) -> number of kept front for neighborhood checking
        - check_limited_neighbors (bool) -> checking a limited neighbors when local search?
        - neighborhood_check_on_potential_sols (bool) -> local search on potential or all solutions?
        - alpha (int, [0, 360]) -> angle for checking knee solution or not
        """
        super().__init__(nas_type='mo')
        self.name = name

        self.k = 3
        self.check_limited_neighbors = True
        self.neighborhood_check_on_potential_sols = True
        self.alpha = 210

        self.footprint = Footprint() if footprint is None else footprint
        self.res_logged = [] if res_logged is None else res_logged

        self.local_archive = ElitistArchive()
        self.archive = ElitistArchive() if archive is None else archive
        self.last_archive = None

        self.S, self.Q = [], []

        self.solutions_collector = None
        self.neighbors_collector = None

        self.last_S_fid, self.last_Q = [], []

        self.list_metrics, self.list_iepochs, self.need_trained = [], [], []
        self.network_history = []

    @property
    def hyperparameters(self):
        return {
            'optimizer': self.name,
            'k': self.k,
            'check_limited_neighbors': self.check_limited_neighbors,
            'neighborhood_check_on_potential_sols': self.neighborhood_check_on_potential_sols,
            'alpha': self.alpha,
        }

    """-------------------------------------------------- SETUP -----------------------------------------------"""
    def setup(self, **kwargs):
        if self.neighborhood_check_on_potential_sols:  # Only performing neighborhood check on knee and extreme ones
            self.solutions_collector = get_potential_solutions
        else:
            self.solutions_collector = get_all_solutions

        if self.check_limited_neighbors:
            self.neighbors_collector = get_some_neighbors
        else:
            self.neighbors_collector = get_all_neighbors

    def _reset(self):
        self.archive = ElitistArchive()
        self.local_archive = ElitistArchive()
        self.last_archive = None
        self.network_history = []

    """------------------------------------------------- EVALUATE --------------------------------------------"""
    def evaluate(self, network, **kwargs):
        list_metrics, max_time = kwargs['list_metrics'], kwargs['max_time']
        train_time, train_epoch, is_terminated = self.problem.mo_evaluate(list_networks=[network],
                                                                          list_metrics=list_metrics,
                                                                          need_trained=self.need_trained,
                                                                          cur_total_time=self.total_time,
                                                                          max_time=max_time)
        self.network_history.append(network)
        self.n_eval += 1
        self.total_time += train_time
        self.total_epoch += train_epoch

    """-------------------------------------------------- SOLVE -----------------------------------------------"""
    def _run(self, **kwargs):
        self._reset()
        self.setup()

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
        first = True
        while not self.isTerminated(max_eval=kwargs['max_eval'], max_time=kwargs['max_time']):  # line 5 - 27
            self.initialize(first, **kwargs)  # Sample new starting solution for the next local search
            first = False
            isContinued = True
            while isContinued:
                isContinued = self.neighborhood_checking(**kwargs)
        return self.archive

    """-------------------------------------------------- UTILITIES -----------------------------------------------"""
    def isTerminated(self, max_eval, max_time):
        if self.n_eval >= max_eval or self.total_time >= max_time:
            return True
        return False

    def update_archive(self, solution):
        self.local_archive.update(solution)
        self.archive.update(solution)

    def initialize(self, first=True, **kwargs):
        start_solution = self.sample_starting_solution(first=first, **kwargs)  # Random a starting solution (line 3)

        # lines 6, 7
        self.S, self.Q = [start_solution], [start_solution]  # approximation set (S) and queue for neighborhood check (Q)
        self.last_archive = deepcopy(self.local_archive)

    def sample_starting_solution(self, first=False, **kwargs):
        if first:
            start_solution = sample_solution(self.footprint.data, self.problem)
        else:
            # lines 16 - 21
            N = []

            ## Choose one elitist in the archive
            available_idx = list(range(len(self.archive.archive)))
            found_new_start = False
            while len(available_idx) != 0:
                idx = np.random.choice(available_idx)
                available_idx.remove(idx)
                selected_solution = deepcopy(self.archive.archive[idx])
                tmp_N, _ = get_all_neighbors(solution=selected_solution, H={}, problem=self.problem)
                N = [neighbor for neighbor in tmp_N if self.problem.get_hash(neighbor) not in self.footprint.data]

                if len(N) != 0:  # If all neighbors of chosen elitist are not visited, choose a random neighbor as new starting solution.
                    found_new_start = True
                    break
            if not found_new_start:  # If not, randomly sampling from the search space.
                start_solution = sample_solution(self.footprint.data, self.problem)
            else:
                idx_selected_neighbor = np.random.choice(len(N))
                start_solution = N[idx_selected_neighbor]

        self.evaluate(start_solution, **kwargs)
        self.update_archive(start_solution)
        return start_solution

    def neighborhood_checking(self, **kwargs):
        N = self.get_neighbors()  # N: neighboring set, line 9

        # lines 10 - 22
        if len(N) == 0:
            # lines 11 - 15
            for fid in range(1, self.k):
                self.Q = self.create_Q(fid=fid)

                N = self.get_neighbors()
                if len(N) != 0:
                    break

            if len(N) == 0:
                return False

        # line 23
        for neighbor in N:
            self.evaluate(neighbor, **kwargs)
            self.update_archive(neighbor)
            if self.isTerminated(max_time=kwargs['max_time'], max_eval=kwargs['max_eval']):
                return False

        # lines 24, 25
        self.create_S(N)

        self.Q = self.create_Q(fid=0)
        return True

    def create_S(self, N):
        P = self.S + N
        F_P = [s.score for s in P]
        idx_fronts = sorter.do(np.array(F_P))
        idx_selected = np.zeros(len(F_P), dtype=bool)
        k = min(len(idx_fronts), self.k)
        for fid in range(k):
            idx_selected[idx_fronts[fid]] = True
            for idx in idx_fronts[fid]:
                P[idx].set('rank', fid)
        self.S = np.array(P)[idx_selected].tolist()
        self.S = remove_duplicate(self.S)

    def create_Q(self, fid):
        Q, last_S_fid, duplicated = self.solutions_collector(S=self.S, fid=fid, alpha=self.alpha,
                                                             last_S_fid=self.last_S_fid, last_Q=self.last_Q,
                                                             problem=self.problem)
        if not duplicated:
            self.last_Q, self.last_S_fid = deepcopy(Q), last_S_fid.copy()

        return Q

    def get_neighbors(self):
        """ Get neighbors of all solutions in queue Q, but discard solutions that has been already in H """
        _H = self.footprint.data
        N = []
        for solution in self.Q:
            tmp_N, _H = self.neighbors_collector(solution, _H, self.problem)

            # Remove duplication
            S_hashes = [self.problem.get_hash(s) for s in self.S]
            N_hashes = [self.problem.get_hash(s) for s in N]
            for neighbor in tmp_N:
                _hash = self.problem.get_hash(neighbor)
                if not_existed(_hash, S=S_hashes, N=N_hashes):
                    N.append(neighbor)
                    N_hashes.append(_hash)
        self.footprint.data = _H
        return N

#####################################################################################
def seeking(list_sol, alpha):
    list_sol = np.array(list_sol)
    non_dominated_front = np.array([solution.score for solution in list_sol])

    ids = range(non_dominated_front.shape[-1])
    info_potential_sols_all = []
    for f_ids in itertools.combinations(ids, 2):
        f_ids = np.array(f_ids)
        obj_1, obj_2 = f'{f_ids[0]}', f'{f_ids[1]}'

        _non_dominated_front = non_dominated_front[:, f_ids].copy()

        ids_sol = np.array(list(range(len(list_sol))))
        ids_fr0 = sorter.do(_non_dominated_front, only_non_dominated_front=True)

        ids_sol = ids_sol[ids_fr0]
        _non_dominated_front = _non_dominated_front[ids_fr0]

        sorted_idx = np.argsort(_non_dominated_front[:, 0])

        ids_sol = ids_sol[sorted_idx]
        _non_dominated_front = _non_dominated_front[sorted_idx]

        min_values, max_values = np.min(_non_dominated_front, axis=0), np.max(_non_dominated_front, axis=0)
        if sum(max_values - min_values) == 0:
            info_potential_sols = [
                [0, list_sol[ids_sol[0]], f'best_f{obj_1}'],  # (idx (in full set), property)
                [0, list_sol[ids_sol[0]], f'best_f{obj_2}']  # (idx (in full set), property),
            ]
            info_potential_sols_all += info_potential_sols
            break
        _non_dominated_front_norm = (_non_dominated_front - min_values) / (max_values - min_values)

        info_potential_sols = [
            [0, list_sol[ids_sol[0]], f'best_f{obj_1}']  # (idx (in full set), property)
        ]

        l_non_front = len(_non_dominated_front)
        for i in range(l_non_front - 1):
            if np.sum(np.abs(_non_dominated_front[i] - _non_dominated_front[i + 1])) != 0:
                break
            else:
                info_potential_sols.append([i + 1, list_sol[ids_sol[i + 1]], f'best_f{obj_1}'])

        for i in range(l_non_front - 1, -1, -1):
            if np.sum(np.abs(_non_dominated_front[i] - _non_dominated_front[i - 1])) != 0:
                break
            else:
                info_potential_sols.append([i - 1, list_sol[ids_sol[i - 1]], f'best_f{obj_2}'])
        info_potential_sols.append([l_non_front - 1, list_sol[ids_sol[l_non_front - 1]], f'best_f{obj_2}'])

        ## find the knee solutions
        start_idx, end_idx = 0, l_non_front - 1

        for i in range(len(info_potential_sols)):
            if info_potential_sols[i + 1][-1] == f'best_f{obj_2}':
                break
            else:
                start_idx = info_potential_sols[i][0] + 1

        for i in range(len(info_potential_sols) - 1, -1, -1):
            if info_potential_sols[i - 1][-1] == f'best_f{obj_1}':
                break
            else:
                end_idx = info_potential_sols[i][0] - 1

        for i in range(start_idx, end_idx + 1):
            l = None
            h = None
            for m in range(i - 1, -1, -1):
                if np.sum(np.abs(_non_dominated_front[m] - _non_dominated_front[i])) != 0:
                    l = m
                    break
            for m in range(i + 1, l_non_front, 1):
                if np.sum(np.abs(_non_dominated_front[m] - _non_dominated_front[i])) != 0:
                    h = m
                    break

            if (h is not None) and (l is not None):
                position = above_or_below(considering_pt=_non_dominated_front[i],
                                          remaining_pt_1=_non_dominated_front[l],
                                          remaining_pt_2=_non_dominated_front[h])
                if position == -1:
                    angle_measure = calc_angle_measure(considering_pt=_non_dominated_front_norm[i],
                                                            neighbor_1=_non_dominated_front_norm[l],
                                                            neighbor_2=_non_dominated_front_norm[h])
                    if angle_measure > alpha:
                        info_potential_sols.append([i, list_sol[ids_sol[i]], 'knee'])
        info_potential_sols_all += info_potential_sols
    return info_potential_sols_all


def above_or_below(considering_pt, remaining_pt_1, remaining_pt_2):
    """
    This function is used to check if the considering point is above or below
    the line connecting two remaining points.\n
    1: above\n
    -1: below
    """
    orthogonal_vector = remaining_pt_2 - remaining_pt_1
    line_connecting_pt1_and_pt2 = -orthogonal_vector[1] * (considering_pt[0] - remaining_pt_1[0]) \
                                  + orthogonal_vector[0] * (considering_pt[1] - remaining_pt_1[1])
    if line_connecting_pt1_and_pt2 > 0:
        return 1
    return -1


def calc_angle_measure(considering_pt, neighbor_1, neighbor_2):
    """
    This function is used to calculate the angle measure is created by the considering point
    and two its nearest neighbors
    """
    line_1 = neighbor_1 - considering_pt
    line_2 = neighbor_2 - considering_pt
    cosine_angle = (line_1[0] * line_2[0] + line_1[1] * line_2[1]) \
                   / (np.sqrt(np.sum(line_1 ** 2)) * np.sqrt(np.sum(line_2 ** 2)))
    if cosine_angle < -1:
        cosine_angle = -1
    if cosine_angle > 1:
        cosine_angle = 1
    angle = np.arccos(cosine_angle)
    return 360 - np.degrees(angle)


def sample_solution(footprint_data, problem):
    while True:
        network = sampling_solution(problem)
        network_hash = problem.get_hash(network)
        if network_hash not in footprint_data:
            return network


def get_all_solutions(S, fid, **kwargs):
    problem = kwargs['problem']
    Q = []
    Q_genotypeHash = []
    rank_S = np.array([s.get('rank') for s in S])
    S_front_i = np.array(S)[rank_S == fid]

    list_genotypeHash = [problem.get_hash(s) for s in S_front_i]
    if is_duplicated(list_genotypeHash, kwargs['last_S_fid']):
        return kwargs['last_Q'], list_genotypeHash, True

    for sol in S_front_i:
        _hash = problem.get_hash(sol)
        if _hash not in Q_genotypeHash:
            Q_genotypeHash.append(_hash)
            Q.append(sol)
    return Q, list_genotypeHash, False


def get_potential_solutions(S, fid, **kwargs):
    alpha = kwargs['alpha']
    problem = kwargs['problem']
    Q = []
    Q_genotypeHash = []
    rank_S = np.array([s.get('rank') for s in S])
    S_front_i = np.array(S)[rank_S == fid]

    list_genotypeHash = [problem.get_hash(s) for s in S_front_i]
    if is_duplicated(list_genotypeHash, kwargs['last_S_fid']):
        return kwargs['last_Q'], list_genotypeHash, True

    info_potential_sols = seeking(S_front_i, alpha)
    potential_sols = [info[1] for info in info_potential_sols]
    for i, sol in enumerate(potential_sols):
        _hash = problem.get_hash(sol)
        if _hash not in Q_genotypeHash:
            Q_genotypeHash.append(_hash)
            Q.append(sol)
    return Q, list_genotypeHash, False


## Get neighboring architectures
def get_some_neighbors(solution, H, problem):
    X, _hash = solution.genotype.copy(), problem.get_hash(solution)
    N = []

    if _hash in H:
        if len(H[_hash]) == 0:
            return [], H
        ids = H[_hash]
        i = np.random.choice(ids)
        H[_hash].remove(i)
    else:
        ids = list(range(len(X)))
        H[_hash] = ids
        i = np.random.choice(H[_hash])
        H[_hash].remove(i)

    _available_ops = problem.search_space.return_available_ops(i).copy()
    _available_ops.remove(X[i])
    for op in _available_ops:
        _X = solution.genotype.copy()
        _X[i] = op
        if problem.search_space.is_valid(_X):
            neighbor = Network()
            neighbor.genotype = _X.copy()
            N.append(neighbor)
    return N, H


def get_all_neighbors(solution, H, problem):
    _hash = problem.get_hash(solution)
    if _hash in H:
        return [], H
    else:
        H[_hash] = []
    N = []

    ids = list(range(len(solution.genotype)))
    X = solution.genotype.copy()
    for i in ids:
        _available_ops = problem.search_space.return_available_ops(i).copy()
        _available_ops.remove(X[i])

        for op in _available_ops:
            _X = solution.genotype.copy()
            _X[i] = op
            if problem.search_space.is_valid(_X):
                neighbor = Network()
                neighbor.genotype = _X.copy()
                N.append(neighbor)
    return N, H

def is_duplicated(genotypeHash_list1, genotypeHash_list2):
    if len(genotypeHash_list1) != len(genotypeHash_list1):
        return False
    for genotypeHash in genotypeHash_list1:
        if genotypeHash not in genotypeHash_list2:
            return False
    return True

########################################################################################################################
def remove_duplicate(pop):
    F = np.array([idv.score for idv in pop])
    is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-8)))[0]
    return np.array(pop)[is_unique].tolist()
