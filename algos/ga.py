from . import Algorithm
from models import Network
import numpy as np
from algos.rea import run_warm_up
from .utils import sampling_solution, update_log
from copy import deepcopy

class GA(Algorithm):
    def __init__(self, crossover_method='2X', prob_crossover=0.9):
        super().__init__()
        self.pop_size = None
        self.tournament_size = None

        self.warm_up = False
        self.metric_warmup = None
        self.n_sample_warmup = 0

        self.prob_mutation = 1.0
        self.crossover_method = crossover_method
        self.prob_crossover = prob_crossover

        self.trend_best_network = []
        self.trend_time = []
        self.network_history = []
        self.score_history = []

        self.pop = []

    def _reset(self):
        self.trend_best_network = []
        self.trend_time = []
        self.network_history = []
        self.score_history = []

        self.pop = []

    def _run(self, **kwargs):
        self._reset()
        if self.warm_up:
            assert self.n_sample_warmup != 0
            assert self.metric_warmup is not None
        assert self.pop_size is not None
        assert self.tournament_size is not None
        max_eval = self.problem.max_eval if self.max_eval is None else self.max_eval
        max_time = self.problem.max_time if self.max_time is None else self.max_time
        if not self.using_zc_metric and self.iepoch is None:
            raise ValueError

        best_network = self.search(max_eval=max_eval, max_time=max_time, metric=self.metric, iepoch=self.iepoch, **kwargs)
        return best_network, self.total_time, self.total_epoch

    def initialize(self, metric, iepoch):
        best_network = Network()
        best_network.score = -np.inf

        tmp_pop = []
        if not self.warm_up:
            for _ in range(self.pop_size):
                network = sampling_solution(problem=self.problem)
                tmp_pop.append(network)
        else:
            tmp_pop, warmup_time = run_warm_up(self.n_sample_warmup, self.pop_size, self.problem, self.metric_warmup)
        for network in tmp_pop:
            info, cost_time = self.evaluate(network, using_zc_metric=self.using_zc_metric, metric=metric, iepoch=iepoch)
            network.score = info[metric]

            self.total_time += cost_time
            self.total_epoch += self.iepoch
            self.trend_time.append(self.total_time)

            if network.score > best_network.score:
                best_network = deepcopy(network)
            update_log(best_network=best_network, cur_network=network, algorithm=self)

            self.pop.append(network)

    def search(self, **kwargs):
        max_eval, max_time = kwargs['max_eval'], kwargs['max_time']
        metric, iepoch = kwargs['metric'], kwargs['iepoch']

        # Initialize population
        self.initialize(metric, iepoch)
        best_network = deepcopy(self.trend_best_network[-1])

        # After the population is seeded, proceed with evolving the population.
        while (self.n_eval < max_eval) and (self.total_time < max_time):
            parents = selection(self.pop, tournament_size=2, n_survive=self.pop_size)

            ## Crossover
            offsprings = crossover(parents=parents, n_offspring=self.pop_size,
                                   prob_crossover=self.prob_crossover, crossover_method=self.crossover_method,
                                   problem=self.problem)
            ## Mutate
            offsprings = mutation(pool=offsprings, prob_mutation=self.prob_mutation, problem=self.problem)

            ## Evaluate
            for network in offsprings:
                info, cost_time = self.evaluate(network, using_zc_metric=self.using_zc_metric, metric=metric, iepoch=iepoch)
                network.score = info[metric]

                self.total_time += cost_time
                self.total_epoch += self.iepoch
                self.trend_time.append(self.total_time)

                if network.score > best_network.score:
                    best_network = deepcopy(network)
                update_log(best_network=best_network, cur_network=network, algorithm=self)

            ## Selection
            pool = parents + offsprings
            self.pop = selection(pool=pool, tournament_size=self.tournament_size, n_survive=self.pop_size)

        return best_network

def mutation(pool, prob_mutation, problem):
    op_mutation_prob = prob_mutation / len(pool[0].genotype)

    mutated_pool = []
    for network in pool:
        n_mutation, max_mutation = 0, 100
        new_network = Network()
        new_network.genotype = network.genotype.copy()
        while n_mutation <= max_mutation:
            n_mutation += 1
            new_genotype = network.genotype.copy()
            for i in range(len(new_genotype)):
                if np.random.random() < op_mutation_prob:
                    available_ops = problem.search_space.return_available_ops(i).copy()
                    _available_ops = [o for o in available_ops if o != new_genotype[i]]
                    new_genotype[i] = np.random.choice(_available_ops)
            if problem.search_space.is_valid(new_genotype):
                new_network.genotype = new_genotype
                break
        mutated_pool.append(new_network)
    return mutated_pool

def crossover(parents, n_offspring, prob_crossover, crossover_method, problem):
    parents = np.array(parents)
    offsprings = []
    while len(offsprings) < n_offspring:
        I = np.random.choice(n_offspring, size=(n_offspring // 2, 2), replace=False)
        parent_pairs = parents[I]
        for pair in parent_pairs:
            if np.random.random() < prob_crossover:
                offspring_genotypes = _crossover(pair[0], pair[1], crossover_method)
                for genotype in offspring_genotypes:
                    if problem.search_space.is_valid(genotype):
                        offspring_net = Network()
                        offspring_net.genotype = genotype
                        offsprings.append(offspring_net)

            else:
                offspring_net1, offspring_net2 = Network(), Network()
                offspring_net1.genotype = pair[0].genotype.copy()
                offspring_net2.genotype = pair[1].genotype.copy()
                offsprings.append(offspring_net1)
                offsprings.append(offspring_net2)
    return offsprings[:n_offspring]

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

def selection(pool, tournament_size, n_survive):
    # Tournament Selection
    pool_survive = []
    pool = np.array(pool)
    while len(pool_survive) < n_survive:
        np.random.shuffle(pool)
        for _ in range(tournament_size):
            I = np.random.permutation(len(pool)).reshape((len(pool) // tournament_size, tournament_size))
            for i in I:
                list_candidates = pool[i]
                list_fitness = [candidate.score for candidate in list_candidates]
                winner = list_candidates[np.argmax(list_fitness)]
                pool_survive.append(winner)
    return pool_survive[:n_survive]

