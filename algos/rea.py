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
        self.metric_warmup = None
        self.zc_metric = None
        self.n_sample_warmup = 0
        self.prob_mutation = 1.0

    def _reset(self):
        self.trend_best_network = []
        self.trend_time = []

    def _run(self, **kwargs):
        best_network = self.search(**kwargs)
        return best_network, self.total_time, self.total_epoch

    def initialize(self, metric):
        best_scores = [-np.inf]
        best_network = None
        population = []  # (validation, spec) tuples

        init_pop = []
        if not self.warm_up:
            list_genotype = []
            for _ in range(self.pop_size):
                while True:
                    genotype = self.problem.search_space.sample(genotype=True)
                    if self.problem.search_space.is_valid(genotype):
                        list_genotype.append(genotype)
                        break
            for genotype in list_genotype:
                network = Network()
                network.genotype = genotype
                init_pop.append(network)
        else:
            init_pop, warmup_time = run_warm_up(self.n_sample_warmup, self.pop_size, self.problem, self.metric_warmup)
        for network in init_pop:
            cost_time = self.evaluate(network, using_zc_metric=self.using_zc_metric, metric=metric)
            self.total_time += cost_time
            self.total_epoch += self.iepoch
            self.trend_time.append(self.total_time)
            population.append((network.score, network.genotype.copy()))

            if network.score > best_scores[-1]:
                best_scores.append(network.score)
                best_network = deepcopy(network)
            self.trend_best_network.append(best_network)
        return population, best_scores, best_network

    def search(self, **kwargs):
        assert self.pop_size is not None
        assert self.tournament_size is not None

        if self.warm_up:
            assert self.n_sample_warmup != 0
            assert self.metric_warmup is not None
        if not self.using_zc_metric:
            metric = self.metric + f'_{self.iepoch}'
        else:
            metric = self.metric
        self._reset()

        # Initialize population
        population, best_scores, best_network = self.initialize(metric)

        # After the population is seeded, proceed with evolving the population.
        while (self.n_eval <= self.problem.max_eval) and (self.total_time <= self.problem.max_time):
            candidates = random_combination(population, self.tournament_size)
            best_candidate = sorted(candidates, key=lambda i: i[0])[-1][1]
            new_network = mutate(best_candidate, self.prob_mutation, problem=self.problem)

            cost_time = self.evaluate(new_network, using_zc_metric=self.using_zc_metric, metric=metric)
            self.total_time += cost_time
            self.total_epoch += self.iepoch
            self.trend_time.append(self.total_time)

            # In regularized evolution, we kill the oldest individual in the population.
            population.append((new_network.score, new_network.genotype.copy()))
            population.pop(0)

            if new_network.score > best_scores[-1]:
                best_scores.append(new_network.score)
                best_network = deepcopy(new_network)
        return best_network

def run_warm_up(n_sample, k, problem, metric):
    list_network, list_scores = [], []
    total_times = 0.0
    for _ in range(n_sample):
        while True:
            genotype = problem.search_space.sample(genotype=True)
            if problem.search_space.is_valid(genotype):
                network = Network()
                network.genotype = genotype
                time = problem.evaluate(network, using_zc_metric=True, metric=metric)
                total_times += time
                list_network.append(network)
                list_scores.append(network.score)
                break
    list_network = np.array(list_network)
    list_scores = np.array(list_scores)
    ids = np.flip(np.argsort(list_scores))
    list_network = list_network[ids]
    return list_network[:k], total_times

def mutate(cur_network_genotype, mutation_rate=1.0, **kwargs):
    problem = kwargs['problem']

    op_mutation_prob = mutation_rate / len(cur_network_genotype)
    n_mutation, max_mutation = 0, 100
    new_network = Network()

    while n_mutation <= max_mutation:
        new_genotype = cur_network_genotype.copy()
        for ind in range(len(new_genotype)):
            if random.random() < op_mutation_prob:
                available_ops = problem.search_space.return_available_ops(ind).copy()
                available = [o for o in available_ops if o != new_genotype[ind]]
                new_genotype[ind] = random.choice(available)
        if problem.search_space.is_valid(new_genotype):
            new_network.genotype = new_genotype
            return new_network
        n_mutation += 1
    new_network.genotype = cur_network_genotype.copy()
    return new_network

def random_combination(iterable, sample_size):
    """Random selection from itertools.combinations(iterable, r)."""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)