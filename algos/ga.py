from . import Algorithm
from models import Network
import numpy as np
from algos.rea import run_warm_up
from .utils import sampling_solution, update_log
from copy import deepcopy

def select_parents(pop):
    parents_pool = []
    for _ in range(2):
        index_pool = np.random.permutation(len(pop)).reshape((len(pop) // 2, 2))
        for idx in index_pool:
            competitor_1, competitor_2 = pop[idx[0]], pop[idx[1]]
            winner = compare(competitor_1, competitor_2)
            parents_pool.append(winner)
    return np.array(parents_pool)

def tournament_selection(pop, tournament_size):
    pool = []
    for _ in range(tournament_size):
        pass


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
        metric = self.metric + f'_{self.iepoch}' if not self.using_zc_metric else self.metric

        best_network = self.search(max_time=max_time, max_eval=max_eval, metric=metric, **kwargs)
        return best_network, self.total_time, self.total_epoch

    def initialize(self, metric):
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
            cost_time = self.evaluate(network, using_zc_metric=self.using_zc_metric, metric=metric)
            self.total_time += cost_time
            self.total_epoch += self.iepoch
            self.trend_time.append(self.total_time)

            update_log(best_network=best_network, cur_network=network, algorithm=self)

            self.pop.append(network)
            self.trend_best_network.append(best_network)

    def search(self, **kwargs):
        max_eval = kwargs['max_eval']
        max_time = kwargs['max_time']
        metric = kwargs['metric']

        # Initialize population
        self.initialize(metric)

        for network in self.pop:
            print(network.genotype, network.score)

        best_network = deepcopy(self.trend_best_network[-1])
        # After the population is seeded, proceed with evolving the population.
        while (self.n_eval <= max_eval) and (self.total_time <= max_time):
            break
        #     candidates = random_combination(population, self.tournament_size)
        #     best_candidate = sorted(candidates, key=lambda i: i[0])[-1][1]
        #     new_network = mutate(best_candidate, self.prob_mutation, problem=self.problem)
        #
        #     time = self.problem.evaluate(new_network, using_zc_metric=self.using_zc_metric, metric=metric)
        #     n_eval += 1
        #     total_time += time
        #     total_epoch += self.iepoch
        #     self.trend_time.append(total_time)
        #
        #     # In regularized evolution, we kill the oldest individual in the population.
        #     population.append((new_network.score, new_network.genotype.copy()))
        #     population.pop(0)
        #
        #     if new_network.score > best_scores[-1]:
        #         best_scores.append(new_network.score)
        #         best_network = deepcopy(new_network)
        return best_network

    def mating(self):
        pass

    def selection(self):
        pass

