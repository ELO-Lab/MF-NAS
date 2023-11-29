"""
Modified from: https://github.com/msu-coinlab/pymoo
"""
import numpy as np

from utils import ElitistArchive, check_not_exist
from .utils import sampling_solution

from . import Algorithm
from models import Network
from copy import deepcopy

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort

class NSGA2(Algorithm):
    def __init__(self, crossover_method='2X', prob_crossover=0.9):
        super().__init__()

        self.pop_size = None

        self.archive = ElitistArchive()
        self.search_log = []

        self.prob_mutation = 1.0
        self.crossover_method = crossover_method
        self.prob_crossover = prob_crossover

        self.metrics = []
        self.iepochs = []
        self.using_zc_metrics = []
        self.pop = []
        self.n_gen = 0

        self.survival = RankAndCrowdingSurvival()

        self.best_search_arch = None
        self.best_test_arch = None
        self.fitness_search_arch = None
        self.fitness_test_arch = None
        self.fitness_test_arch1 = None

    def evaluate(self, network, using_zc_metrics, metrics, iepochs):
        scores = []
        total_cost_time = 0.0
        for i in range(len(metrics)):
            info, cost_time = self.problem.evaluate(network, using_zc_metric=using_zc_metrics[i],
                                                    metric=metrics[i], iepoch=iepochs[i])
            score = info[metrics[i]]
            total_cost_time += cost_time
            scores.append(score)
        self.n_eval += 1

        # if self.n_eval == 1:
        #     search_F = np.round((1 - scores[0]) * 100, 2)
        #     test_F = self.problem.get_test_performance(network)[0]
        #     self.best_search_arch = network.genotype.copy()
        #     self.best_test_arch = network.genotype.copy()
        #     self.fitness_search_arch = search_F
        #     self.fitness_test_arch = test_F
        #     self.fitness_test_arch1 = test_F
        # else:
        #     search_F = np.round((1 - scores[0]) * 100, 2)
        #     test_F = self.problem.get_test_performance(network)[0]
        #     if search_F > self.fitness_search_arch:
        #         self.best_search_arch = network.genotype.copy()
        #         self.fitness_search_arch = search_F
        #         self.fitness_test_arch = test_F
        #     if test_F > self.fitness_test_arch1:
        #         self.best_test_arch = network.genotype.copy()
        #         self.fitness_test_arch1 = test_F
        # self.search_log.append(
        #     [''.join(list(map(str, self.best_search_arch))), self.fitness_search_arch, self.fitness_test_arch, ''.join(list(map(str, self.best_test_arch))),
        #      self.fitness_test_arch1])

        return scores, total_cost_time

    def _reset(self):
        self.archive = ElitistArchive()
        self.search_log = []
        self.pop = []
        self.n_gen = 0

        self.best_search_arch = None
        self.best_test_arch = None
        self.fitness_search_arch = None
        self.fitness_test_arch = None
        self.fitness_test_arch1 = None

    def _run(self, **kwargs):
        self._reset()
        max_eval = self.problem.max_eval if self.max_eval is None else self.max_eval
        max_time = self.problem.max_time if self.max_time is None else self.max_time

        approximation_set = self.search(max_eval=max_eval, max_time=max_time, metric=self.metric, iepoch=self.iepoch, **kwargs)
        return approximation_set, self.total_time, self.total_epoch

    def search(self, **kwargs):
        max_eval, max_time = kwargs['max_eval'], kwargs['max_time']

        self.initialize()
        self.pop = self.survival.do(self.pop, self.pop_size)
        while (self.n_eval < max_eval) and (self.total_time < max_time):
            self.n_gen += 1
            self.next()

        list_network = []
        for i in range(len(self.archive.genotype)):
            network = Network()
            network.set(['genotype', 'ID', 'score'],
                        [self.archive.genotype[i], self.archive.ID[i], self.archive.fitness[i]])
            list_network.append(network)
        return list_network

    def initialize(self):
        for _ in range(self.pop_size):
            network = sampling_solution(problem=self.problem)
            network.set('ID', ''.join(list(map(str, network.genotype))))

            scores, cost_time = self.evaluate(network, using_zc_metrics=self.using_zc_metrics,
                                              metrics=self.metrics, iepochs=self.iepochs)
            network.set('score', scores)
            self.archive.update(sol=network, algorithm=self)

            self.total_time += cost_time
            self.total_epoch += max(self.iepochs)

            self.pop.append(network)

    def mating(self):
        # Selection
        parents = binary_tournament_selection(self.pop)

        # Crossover
        offsprings = crossover(parents=parents, n_offspring=self.pop_size,
                               prob_crossover=self.prob_crossover, crossover_method=self.crossover_method,
                               problem=self.problem)
        # Mutate
        offsprings = mutation(pool=offsprings, pop=self.pop, prob_mutation=self.prob_mutation, problem=self.problem)

        # Evaluate
        for network in offsprings:
            scores, cost_time = self.evaluate(network, using_zc_metrics=self.using_zc_metrics,
                                              metrics=self.metrics, iepochs=self.iepochs)
            network.set('score', scores)
            self.archive.update(sol=network, algorithm=self)

            self.total_time += cost_time
            self.total_epoch += max(self.iepochs)
        return offsprings

    def next(self):
        offsprings = self.mating()
        pool = self.pop + offsprings
        self.pop = self.survival.do(pool, self.pop_size)


def mutation(pool, pop, prob_mutation, problem):
    op_mutation_prob = prob_mutation / len(pool[0].genotype)
    pop_ID = [sol.get('ID') for sol in pop]
    for network in pool:
        while True:
            new_genotype = network.genotype.copy()
            for m, prob in enumerate(np.random.rand(len(pool[0].genotype))):
                if prob <= op_mutation_prob:
                    available_ops = problem.search_space.return_available_ops(m).copy()
                    available_ops.remove(new_genotype[m])
                    new_genotype[m] = np.random.choice(available_ops)
            if problem.search_space.is_valid(new_genotype):
                new_ID = ''.join(list(map(str, new_genotype)))
                if check_not_exist(new_ID, pop_ID=pop_ID):
                    network.set(['genotype', 'ID'], [new_genotype, new_ID])
                    pop_ID.append(new_ID)
                    break
    return pool

def crossover(parents, n_offspring, prob_crossover, crossover_method, problem):
    parents = np.array(parents)
    offsprings = []
    offsprings_ID = []

    n = 0
    n_crossover, max_crossover = 0, n_offspring * 5

    while True:
        I = np.random.choice(n_offspring, size=(n_offspring // 2, 2), replace=False)
        parent_pairs = parents[I]
        for pair in parent_pairs:
            if np.random.random() < prob_crossover:
                offspring_genotypes = _crossover(pair[0], pair[1], crossover_method)
                for genotype in offspring_genotypes:
                    if problem.search_space.is_valid(genotype):
                        ID = ''.join(list(map(str, genotype)))
                        if check_not_exist(ID, offsprings_ID=offsprings_ID) or (n_crossover - max_crossover > 0):
                            offspring_net = Network()
                            offspring_net.genotype = genotype
                            offspring_net.set('ID', ID)

                            offsprings.append(offspring_net)
                            offsprings_ID.append(ID)
                            n += 1
                            if n - n_offspring >= 0:
                                return offsprings[:n_offspring]

            else:
                offspring_net1, offspring_net2 = deepcopy(pair[0]), deepcopy(pair[1])
                offsprings.append(offspring_net1)
                offsprings.append(offspring_net2)
                offsprings_ID.append(''.join(list(map(str, offspring_net1.genotype))))
                offsprings_ID.append(''.join(list(map(str, offspring_net2.genotype))))
                n += 2
                if n - n_offspring >= 0:
                    return offsprings[:n_offspring]
        n_crossover += 1

def _crossover(parent_1, parent_2, crossover_method):
    genotype_1, genotype_2 = parent_1.genotype.copy(), parent_2.genotype.copy()

    if crossover_method == '1X':  # 1-point crossover
        i = np.random.randint(1, len(genotype_1))
        genotype_1[i:], genotype_2[i:] = genotype_2[i:], genotype_1[i:].copy()

    elif crossover_method == '2X':  # 2-point crossover
        I = np.random.choice(range(1, len(genotype_1) - 1), 2, replace=False)
        i_1, i_2 = min(I), max(I)

        genotype_1[i_1: i_2], genotype_2[i_1: i_2] = genotype_2[i_1: i_2], genotype_1[i_1: i_2].copy()

    elif crossover_method == 'UX':  # Uniform crossover
        I = np.random.randint(0, 2, genotype_1.shape, dtype=bool)

        genotype_1[I], genotype_2[I] = genotype_2[I], genotype_1[I].copy()

    return [genotype_1, genotype_2]


def x_better_y(x, y):
    r_x, r_y = x.get('rank'), y.get('rank')
    if r_x < r_y:
        return x
    elif r_x > r_y:
        return y
    else:
        cd_x, cd_y = x.get('crowding'), y.get('crowding')
        if cd_x > cd_y:
            return x
        elif cd_x < cd_y:
            return y
        else:
            return x

def binary_tournament_selection(pool):
    # Binary Tournament Selection
    pool_survive = []
    pool = np.array(pool)
    for _ in range(2):
        I = np.random.permutation(len(pool)).reshape((len(pool) // 2, 2))
        for i in I:
            candidate_1, candidate_2 = pool[i[0]], pool[i[1]]
            winner = x_better_y(x=candidate_1, y=candidate_2)
            pool_survive.append(winner)
    return np.array(pool_survive)


class RankAndCrowdingSurvival:
    def __init__(self):
        self.name = 'Rank and Crowding Survival'

    @staticmethod
    def do(pop, n_survive):
        # get the objective space values and objects
        F = np.array([sol.get('score') for sol in pop])

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calculating_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set('rank', k)
                pop[i].set('crowding', crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])
        return np.array(pop)[survivors].tolist()


def calculating_crowding_distance(F):
    infinity = 1e+14

    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, infinity)
    else:

        # sort each column and get index
        I = np.argsort(F, axis=0, kind='mergesort')

        # now really sort the whole array
        F = F[I, np.arange(n_obj)]

        # get the distance to the last element in sorted list and replace zeros with actual values
        dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) - np.concatenate([np.full((1, n_obj), -np.inf), F])

        index_dist_is_zero = np.where(dist == 0)

        dist_to_last = np.copy(dist)
        for i, j in zip(*index_dist_is_zero):
            dist_to_last[i, j] = dist_to_last[i - 1, j]

        dist_to_next = np.copy(dist)
        for i, j in reversed(list(zip(*index_dist_is_zero))):
            dist_to_next[i, j] = dist_to_next[i + 1, j]

        # normalize all the distances
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        norm[norm == 0] = np.nan
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divided by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    # replace infinity with a large number
    crowding[np.isinf(crowding)] = infinity
    return crowding