import numpy as np
from models import Network
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from algos.utils import not_existed

############################## NON-DOMINATED SORTING RANK AND CROWING DISTANCE SELECTION ##############################
class RankAndCrowdingSurvival:
    def __init__(self):
        self.name = 'Rank and Crowding Survival'

    @staticmethod
    def do(pop, n_survive):
        # get the objective space values and objects
        F = np.array([network.get('score') for network in pop])
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
        pop = np.array(pop)
        return pop[survivors].tolist()


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

############################################# INTEGER-ENCODING MUTATION #############################################
class BitStringMutation:
    def __init__(self, prob=1.0):
        self.prob = prob

    def do(self, problem, P, O, **kwargs):
        P_hashes = [''.join(map(str, network.genotype)) for network in P]
        current_O_genotypes = np.array([network.genotype for network in P])

        offspring_size = len(O)
        l = len(current_O_genotypes[-1])

        nMutations, maxMutations = 0, offspring_size * 5

        self.prob = 1/l

        new_O = []
        new_O_hashes = []

        n = 0
        while True:
            for genotype in current_O_genotypes:
                _genotype = genotype.copy()

                for m, prob in enumerate(np.random.rand(l)):
                    if prob <= self.prob:
                        available_ops = problem.search_space.return_available_ops(m)
                        available_ops.remove(_genotype[m])
                        new_op = np.random.choice(available_ops)
                        _genotype[m] = new_op

                if problem.search_space.is_valid(_genotype):
                    _hash = ''.join(map(str, _genotype))

                    if not_existed(_hash, P=P_hashes) or (nMutations - maxMutations > 0):
                        new_O_hashes.append(_hash)

                        network = Network()
                        network.genotype = _genotype
                        new_O.append(network)
                        n += 1
                        if n - offspring_size == 0:
                            return new_O

            nMutations += 1

#################################################### X-POINT CROSSOVER #################################################
class PointCrossover:
    def __init__(self, prob=0.9, method=None):
        self.n_parents = 2
        available_methods = ['1X', '2X', 'UX']
        if method not in available_methods:
            raise ValueError('Invalid crossover method: ' + method)
        else:
            self.method = method
        self.prob = prob

    def do(self, problem, parents_pool, P, **kwargs):
        offspring_size = len(parents_pool)
        O = []
        O_hashes = []

        n = 0
        nCrossovers, maxCrossovers = 0, offspring_size * 5

        while True:
            I = np.random.choice(offspring_size, size=(offspring_size // 2, self.n_parents), replace=False)
            parents = parents_pool[I]
            for i in range(len(parents)):
                if np.random.random() < self.prob:
                    off_genotypes = crossover(parents[i][0].genotype, parents[i][1].genotype, self.method)
                    for j, _genotype in enumerate(off_genotypes):
                        if problem.search_space.is_valid(_genotype):
                            _hash = ''.join(map(str, _genotype))

                            if not_existed(_hash, O=O_hashes) or (nCrossovers - maxCrossovers > 0):
                                O_hashes.append(_hash)
                                network = Network()
                                network.genotype = _genotype
                                O.append(network)
                                n += 1
                                if n - offspring_size == 0:
                                    return O
                else:
                    for network in parents[i]:
                        _hash = ''.join(map(str, network.genotype))
                        O_hashes.append(_hash)
                        O.append(network)
                        n += 1
                        if n - offspring_size == 0:
                            return O
            nCrossovers += 1

def crossover(parent_1, parent_2, typeC, **kwargs):
    offspring_1, offspring_2 = parent_1.copy(), parent_2.copy()

    if typeC == '1X':  # 1-point crossover
        point = np.random.randint(1, len(parent_1))

        offspring_1[point:], offspring_2[point:] = offspring_2[point:], offspring_1[point:].copy()

    elif typeC == '2X':  # 2-point crossover
        points_list = np.random.choice(range(1, len(parent_1) - 1), 2, replace=False)

        low_idx, up_idx = min(points_list), max(points_list)

        offspring_1[low_idx: up_idx], offspring_2[low_idx: up_idx] = \
            offspring_2[low_idx: up_idx], offspring_1[low_idx: up_idx].copy()

    elif typeC == 'UX':  # Uniform crossover
        pts = np.random.randint(0, 2, parent_1.shape, dtype=bool)

        offspring_1[pts], offspring_2[pts] = offspring_2[pts], offspring_1[pts].copy()

    return [offspring_1, offspring_2]
