import numpy as np
from copy import deepcopy
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from models import Network
from utils import ElitistArchive, check_not_exist
from . import Algorithm

class LOMONAS(Algorithm):
    """
    LOMONAS - Local search algorithm for Multi-objective Neural Architecture Search
    """

    def __init__(self):
        super().__init__()

        self.k = 3

        self.archive = ElitistArchive()
        self.search_log = []

        self.metrics = []
        self.iepochs = []
        self.using_zc_metrics = []
        self.weighted = []

    def evaluate(self, network, using_zc_metrics, metrics, iepochs):
        scores = []
        total_cost_time = 0.0
        for i in range(len(metrics)):
            info, cost_time = self.problem.evaluate(network, using_zc_metric=using_zc_metrics[i],
                                                    metric=metrics[i], iepoch=iepochs[i])
            score = info[metrics[i]] * self.weighted[i]
            total_cost_time += cost_time
            scores.append(score)
        self.n_eval += 1
        return scores, total_cost_time

    def _reset(self):
        self.archive = ElitistArchive()
        self.search_log = []

    def _run(self, **kwargs):
        self._reset()
        max_eval = self.problem.max_eval if self.max_eval is None else self.max_eval
        max_time = self.problem.max_time if self.max_time is None else self.max_time

        approximation_set = self.search(max_eval=max_eval, max_time=max_time, metric=self.metric, iepoch=self.iepoch, **kwargs)
        return approximation_set, self.total_time, self.total_epoch

    def search(self, **kwargs):
        max_eval, max_time = kwargs['max_eval'], kwargs['max_time']
        footprint = {}

        # lines 3 - 4
        init_sol = sample(footprint, self.problem)
        scores, cost_time = self.evaluate(init_sol, using_zc_metrics=self.using_zc_metrics,
                                          metrics=self.metrics, iepochs=self.iepochs)

        init_sol.set('score', scores)

        self.total_time += cost_time
        self.total_epoch += max(self.iepochs)

        starting_sol_genotype = init_sol.genotype
        starting_sol_ID = init_sol.get('ID')
        starting_sol_fitness = init_sol.get('score')

        while True:
            starting_network = Network()
            starting_network.set(['genotype', 'ID', 'score'], [starting_sol_genotype, starting_sol_ID, starting_sol_fitness])
            self.archive.update(sol=starting_network, algorithm=self)

            S_genotype, S_ID, S_fitness = [starting_network.genotype], [starting_network.get('ID')], [starting_network.get('score')]  # line 6
            Q_genotype = [starting_network.genotype]  # line 7

            while True:
                N_genotype, N_ID, footprint = self.get_neighbors(Q_genotype=Q_genotype,
                                                                footprint=footprint, S_ID=S_ID)  # line 9
                N_fitness = []
                if len(N_genotype) == 0:  # line 10
                    # lines 11 - 15
                    for fid in range(2, self.k + 1):
                        Q_genotype = get_potential_solutions(S_genotype, S_fitness, fid)

                        N_genotype, N_ID, footprint = self.get_neighbors(Q_genotype=Q_genotype,
                                                                         footprint=footprint, S_ID=S_ID)
                        if len(N_genotype) != 0:
                            break

                    # lines 16 - 21
                    if len(N_genotype) == 0:
                        A_genotype = self.archive.genotype
                        while True:
                            selected_genotype = A_genotype[np.random.choice(len(A_genotype))]
                            N_genotype = []
                            _N_genotype = _get_all_neighbors(self.problem, selected_genotype)

                            for _genotype in _N_genotype:
                                if self.problem.search_space.is_valid(_genotype):
                                    # _ID = ''.join(list(map(str, _genotype)))
                                    _ID = self.problem.get_h(_genotype)
                                    if _ID not in footprint:
                                        N_genotype.append(_genotype)
                            if len(N_genotype) != 0:
                                break

                        sol = Network()
                        starting_sol_genotype = N_genotype[np.random.choice(len(N_genotype))]
                        starting_sol_ID = ''.join(list(map(str, starting_sol_genotype)))

                        sol.set('genotype', starting_sol_genotype)
                        scores, cost_time = self.evaluate(sol, using_zc_metrics=self.using_zc_metrics,
                                                          metrics=self.metrics, iepochs=self.iepochs)
                        self.total_time += cost_time
                        self.total_epoch += max(self.iepochs)

                        starting_sol_fitness = scores
                        break

                # lines 23
                for genotype in N_genotype:
                    network = Network()
                    ID = ''.join(list(map(str, genotype)))
                    network.set(['genotype', 'ID'], [genotype, ID])

                    scores, cost_time = self.evaluate(network, using_zc_metrics=self.using_zc_metrics,
                                                      metrics=self.metrics, iepochs=self.iepochs)
                    network.set('score', scores)

                    self.total_time += cost_time
                    self.total_epoch += max(self.iepochs)

                    N_fitness.append(network.get('score'))
                    self.archive.update(network, algorithm=self)

                # line 24
                P_genotype = S_genotype + N_genotype
                P_ID = S_ID + N_ID
                P_fitness = S_fitness + N_fitness

                I_front = NonDominatedSorting().do(np.array(P_fitness))
                I_selected = np.zeros(len(P_fitness), dtype=bool)
                N = min(len(I_front), self.k)
                for i in range(N):
                    for j in I_front[i]:
                        I_selected[j] = True

                S_genotype = np.array(deepcopy(P_genotype))[I_selected].tolist()
                S_ID = np.array(deepcopy(P_ID))[I_selected].tolist()
                S_fitness = np.array(deepcopy(P_fitness))[I_selected].tolist()

                # line 25
                Q_genotype = get_potential_solutions(S_genotype, S_fitness, 1)

                if self.n_eval >= max_eval or self.total_time >= max_time:
                    list_network = []
                    for i in range(len(self.archive.genotype)):
                        network = Network()
                        network.set(['genotype', 'ID', 'score'], [self.archive.genotype[i], self.archive.ID[i], self.archive.fitness[i]])
                        list_network.append(network)
                    return list_network

    ########################################################################################################## Utilities
    def get_neighbors(self, Q_genotype, footprint, S_ID):
        _footprint = footprint
        neighbor_genotype, neighbor_ID = [], []

        for genotype in Q_genotype:
            _neighbor_genotype, _footprint = get_partial_neighbors(genotype, _footprint, self.problem)

            for _genotype in _neighbor_genotype:
                if self.problem.search_space.is_valid(_genotype):
                    _ID = ''.join(list(map(str, _genotype)))
                    if check_not_exist(_ID, S_ID=S_ID, neighbors_ID=neighbor_ID):
                        neighbor_genotype.append(_genotype)
                        neighbor_ID.append(_ID)
        return neighbor_genotype, neighbor_ID, _footprint


#####################################################################################
def seeking(X_list, F_list):
    non_dominated_set = X_list.copy()
    non_dominated_front = F_list.copy()

    sorted_idx = np.argsort(non_dominated_front[:, 0])

    non_dominated_set = non_dominated_set[sorted_idx]
    non_dominated_front = non_dominated_front[sorted_idx]

    non_dominated_front_norm = non_dominated_front.copy()

    min_f0 = np.min(non_dominated_front[:, 0])
    max_f0 = np.max(non_dominated_front[:, 0])

    min_f1 = np.min(non_dominated_front[:, 1])
    max_f1 = np.max(non_dominated_front[:, 1])

    non_dominated_front_norm[:, 0] = (non_dominated_front_norm[:, 0] - min_f0) / (max_f0 - min_f0)
    non_dominated_front_norm[:, 1] = (non_dominated_front_norm[:, 1] - min_f1) / (max_f1 - min_f1)

    potential_sols = [
        [0, non_dominated_set[0], 'best_f0']  # (idx (in full set), property)
    ]

    for i in range(len(non_dominated_front) - 1):
        if np.sum(np.abs(non_dominated_front[i] - non_dominated_front[i + 1])) != 0:
            break
        else:
            potential_sols.append([i + 1, non_dominated_set[i + 1], 'best_f0'])

    for i in range(len(non_dominated_front) - 1, -1, -1):
        if np.sum(np.abs(non_dominated_front[i] - non_dominated_front[i - 1])) != 0:
            break
        else:
            potential_sols.append([i - 1, non_dominated_set[i - 1], 'best_f1'])
    potential_sols.append([len(non_dominated_front) - 1, non_dominated_set[len(non_dominated_front) - 1], 'best_f1'])

    ## find the knee solutions
    start_idx = potential_sols[0][0]
    end_idx = potential_sols[-1][0]

    for i in range(len(potential_sols)):
        if potential_sols[i + 1][-1] == 'best_f1':
            break
        else:
            start_idx = potential_sols[i][0] + 1

    for i in range(len(potential_sols) - 1, -1, -1):
        if potential_sols[i - 1][-1] == 'best_f0':
            break
        else:
            end_idx = potential_sols[i][0] - 1

    for i in range(start_idx, end_idx + 1):
        l = None
        h = None
        for m in range(i - 1, -1, -1):
            if np.sum(np.abs(non_dominated_front[m] - non_dominated_front[i])) != 0:
                l = m
                break
        for m in range(i + 1, len(non_dominated_front), 1):
            if np.sum(np.abs(non_dominated_front[m] - non_dominated_front[i])) != 0:
                h = m
                break

        if (h is not None) and (l is not None):
            position = check_above_or_below(considering_pt=non_dominated_front[i],
                                            remaining_pt_1=non_dominated_front[l],
                                            remaining_pt_2=non_dominated_front[h])
            if position == -1:
                angle_measure = calculate_angle_measure(considering_pt=non_dominated_front_norm[i],
                                                        neighbor_1=non_dominated_front_norm[l],
                                                        neighbor_2=non_dominated_front_norm[h])
                if angle_measure > 210:
                    potential_sols.append([i, non_dominated_set[i], 'knee'])

    return potential_sols


def check_above_or_below(considering_pt, remaining_pt_1, remaining_pt_2):
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


def calculate_angle_measure(considering_pt, neighbor_1, neighbor_2):
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

def sample(footprint, problem):
    network = Network()
    while True:
        genotype = problem.search_space.sample(genotype=True)
        # ID = ''.join(list(map(str, genotype)))
        if problem.search_space.is_valid(genotype):
            ID = problem.get_h(genotype)
            if ID not in footprint:
                network.set('genotype', genotype)
                network.set('ID', ID)
                return network

def get_partial_neighbors(genotype, footprint, problem):
    # ID = ''.join(list(map(str, genotype)))
    ID = problem.get_h(genotype)

    if ID in footprint:
        if len(footprint[ID]) == 0:
            return [], footprint
        I = footprint[ID]
        i = np.random.choice(I)
        footprint[ID].remove(i)
    else:
        I = list(range(len(genotype)))
        footprint[ID] = I
        i = np.random.choice(footprint[ID])
        footprint[ID].remove(i)

    OPS = problem.search_space.return_available_ops(i).copy()
    neighbor_genotype = []
    for op in OPS:
        _genotype = np.array(genotype.copy())
        _genotype[i] = op
        neighbor_genotype.append(_genotype)
    return neighbor_genotype, footprint


####################################### Get architectures for local search #############################################
## Get potential architectures (knee and extreme ones)
def get_potential_solutions(list_genotype, list_fitness, NF):
    potential_genotypes = []

    I_fronts = NonDominatedSorting().do(np.array(list_fitness))
    N = min(len(I_fronts), NF)
    for i in range(N):
        I_selected = np.zeros(len(list_fitness), dtype=bool)
        I_selected[I_fronts[i]] = True

        front_i_genotype = np.array(list_genotype)[I_selected]
        front_i_fitness = np.array(list_fitness)[I_selected]

        _potential_sols = seeking(front_i_genotype, front_i_fitness)
        potential_sols = np.array([info[1] for info in _potential_sols])

        for genotype in potential_sols:
            potential_genotypes.append(genotype)
    return potential_genotypes

## Get all neighbors (for creating the neighbors dictionary)
def _get_all_neighbors(problem, genotype):
    neighbor_genotype = []
    I = list(range(len(genotype)))

    for i in I:
        OPS = problem.search_space.return_available_ops(i).copy()
        OPS.remove(genotype[i])
        for op in OPS:
            _genotype = genotype.copy()
            _genotype[i] = op
            neighbor_genotype.append(_genotype)
    return neighbor_genotype