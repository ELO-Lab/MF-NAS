import random
import numpy as np
from copy import deepcopy
from models import Network
from algos import Algorithm
from algos.utils import update_log, sampling_solution

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
        self.network_history = []
        self.score_history = []
        self.best_network = None

    def _reset(self):
        self.trend_best_network = []
        self.trend_time = []
        self.network_history = []
        self.score_history = []
        self.best_network = None

    def _run(self, **kwargs):
        max_eval = self.problem.max_eval if self.max_eval is None else self.max_eval
        max_time = self.problem.max_time if self.max_time is None else self.max_time
        metric = self.metric + f'_{self.iepoch}' if not self.using_zc_metric else self.metric

        best_network = self.search(max_eval=max_eval, max_time=max_time, metric=metric, **kwargs)
        return best_network, self.total_time, self.total_epoch

    def evaluate(self, network, metric=None, using_zc_metric=None):
        if using_zc_metric is None:
            using_zc_metric = self.using_zc_metric
        if metric is None:
            metric = self.search_metric
        total_time, total_epoch, is_terminated = self.problem.evaluate(network,
                                                                       metric=metric, using_zc_metric=using_zc_metric,
                                                                       cur_total_time=self.total_time,
                                                                       max_time=self.max_time)
        self.n_eval += 1
        self.total_time += total_time
        self.total_epoch += total_epoch
        return is_terminated or self.n_eval >= self.max_eval

    def initialize(self):
        init_pop = []
        if not self.warm_up:
            list_hashes = []
            for _ in range(self.pop_size):
                while True:
                    network = sampling_solution(self.problem)
                    network_hash = self.problem.get_hash(network)
                    if network_hash not in list_hashes:
                        list_hashes.append(network_hash)
                        init_pop.append(network)
                        break
        else:
            init_pop = run_warm_up(algo=self)
        return init_pop

    def search(self, **kwargs):
        self._reset()
        self.max_eval = kwargs['max_eval']
        self.max_time = kwargs['max_time']
        self.search_metric = kwargs['metric']

        assert self.pop_size is not None
        assert self.tournament_size is not None

        if self.warm_up:
            assert self.n_sample_warmup != 0
            assert self.metric_warmup is not None

        population = []  # (validation, spec) tuples

        # Initialize population
        samples = self.initialize()
        for network in samples:
            is_terminated = self.evaluate(network)
            population.append((network.score, network.genotype.copy()))

            if self.best_network is None:
                self.best_network = deepcopy(network)
            else:
                if network.score > self.best_network.score:
                    self.best_network = deepcopy(network)
            update_log(best_network=self.best_network, cur_network=network, algorithm=self)
            if is_terminated:
                return self.best_network

        # After the population is seeded, proceed with evolving the population.
        while True:
            candidates = random_combination(population, self.tournament_size)
            best_candidate = sorted(candidates, key=lambda i: i[0])[-1][1]
            new_network = mutate(best_candidate, self.prob_mutation, problem=self.problem)

            is_terminated = self.evaluate(new_network)

            # In regularized evolution, we kill the oldest individual in the population.
            population.append((new_network.score, new_network.genotype.copy()))
            population.pop(0)

            if new_network.score > self.best_network.score:
                self.best_network = deepcopy(new_network)
            update_log(best_network=self.best_network, cur_network=new_network, algorithm=self)
            if is_terminated:
                return self.best_network

def run_warm_up(algo):
    n_sample = algo.n_sample_warmup
    k = algo.pop_size
    problem = algo.problem
    metric = algo.metric_warmup

    list_network, list_scores = [], []
    pop_hashes = []
    for _ in range(n_sample):
        while True:
            network = sampling_solution(problem)
            network_hash = problem.get_hash(network)
            if network_hash not in pop_hashes:
                pop_hashes.append(network_hash)
                is_terminated = algo.evaluate(network, using_zc_metric=True, metric=metric)
                algo.n_eval = 0
                list_network.append(network)
                list_scores.append(network.score)
                if is_terminated:
                    break
                break
    list_network = np.array(list_network)
    list_scores = np.array(list_scores)
    ids = np.flip(np.argsort(list_scores))
    list_network = list_network[ids]
    return list_network[:k]

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
