from models import Network
from algos import Algorithm
import numpy as np
from copy import deepcopy
import random

class REA(Algorithm):
    def __init__(self):
        super().__init__()
        self.trend_best_network = []
        self.trend_time = []
        self.pop_size = None
        self.tournament_size = None
        self.warm_up = False
        self.zc_metric = None
        self.n_sample_warmup = 0
        self.prob_mutation = 1.0

    def _reset(self):
        self.trend_best_network = []
        self.trend_time = []

    def _run(self, **kwargs):
        assert self.pop_size is not None
        assert self.tournament_size is not None
        self.pop_size = int(self.pop_size)
        self.tournament_size = int(self.tournament_size)
        self.prob_mutation = float(self.prob_mutation)

        if self.warm_up:
            assert self.n_sample_warmup != 0
            assert self.zc_metric is not None
        self._reset()

        n_eval = 0
        total_time = 0

        times, best_scores = [0.0], [-np.inf]
        best_network = None
        population = []  # (validation, spec) tuples

        # For the first population_size individuals, seed the population with randomly
        # generated cells.

        init_pop = []
        if not self.warm_up:
            list_genotype = [self.problem.search_space.sample(genotype=True) for _ in range(self.pop_size)]
            for genotype in list_genotype:
                network = Network()
                network.genotype = genotype
                init_pop.append(network)
        else:
            init_pop = run_warm_up(self.n_sample_warmup, self.pop_size, self.problem, self.zc_metric)
        for network in init_pop:
            time = self.problem.evaluate(network, algorithm=self)
            n_eval += 1
            total_time += time
            self.trend_time.append(total_time)
            population.append((network.score, network.genotype.copy()))

            if network.score > best_scores[-1]:
                best_scores.append(network.score)
                best_network = deepcopy(network)
            self.trend_best_network.append(best_network)

        # After the population is seeded, proceed with evolving the population.
        while (n_eval <= self.problem.max_eval) and (total_time <= self.problem.max_time):
            candidates = random_combination(population, self.tournament_size)
            best_candidate = sorted(candidates, key=lambda i: i[0])[-1][1]
            new_network = mutate(best_candidate, self.prob_mutation, available_ops=self.problem.search_space.available_ops)

            time = self.problem.evaluate(new_network, algorithm=self)
            n_eval += 1
            total_time += time
            self.trend_time.append(total_time)

            # In regularized evolution, we kill the oldest individual in the population.
            population.append((new_network.score, new_network.genotype.copy()))
            population.pop(0)

            if new_network.score > best_scores[-1]:
                best_scores.append(new_network.score)
                best_network = deepcopy(new_network)
        return best_network, total_time

def run_warm_up(n_sample, k, problem, zc_metric):
    list_network, list_scores = [], []
    total_times = 0.0
    for _ in range(n_sample):
        genotype = problem.search_space.sample(genotype=True)
        network = Network()
        network.genotype = genotype
        time = problem.evaluate(network, metric=zc_metric)
        total_times += time
        list_network.append(network)
        list_scores.append(network.score)
    list_network = np.array(list_network)
    list_scores = np.array(list_scores)
    ids = np.flip(np.argsort(list_scores))
    list_network = list_network[ids]
    return list_network[:k], total_times

def mutate(cur_network, mutation_rate=1.0, **kwargs):
    available_ops = kwargs['available_ops']
    new_genotype = cur_network.genotype.copy()

    op_mutation_prob = mutation_rate / len(new_genotype)
    for ind in range(len(new_genotype)):
        if random.random() < op_mutation_prob:
            available = [o for o in available_ops if o != new_genotype[ind]]
            new_genotype[ind] = random.choice(available)
    new_network = Network()
    new_network.genotype = new_genotype
    return new_network

def random_combination(iterable, sample_size):
    """Random selection from itertools.combinations(iterable, r)."""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)