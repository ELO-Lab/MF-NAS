import numpy as np
from copy import deepcopy
from algos import Algorithm
from algos.utils import update_log, sampling_solution
from utils.genetic_operators import PointCrossover, BitStringMutation

class GA(Algorithm):
    def __init__(self):
        super().__init__()
        self.trend_best_network = []
        self.trend_time = []
        self.pop_size = None
        self.pop = None
        self.tournament_size = None
        self.warm_up = False
        self.metric_warmup = None
        self.zc_metric = None
        self.n_sample_warmup = 0

        self.n_gen = 0

        self.prob_c, self.prob_m = -1, -1
        self.crossover_method = None
        self.crossover, self.mutation, self.survival = None, None, None

        self.network_history = []
        self.score_history = []

    def get_genetic_operators(self):
        self.crossover = PointCrossover(prob=self.prob_c, method=self.crossover_method)
        self.mutation = BitStringMutation(prob=self.prob_m)

    def _reset(self):
        self.pop = None

        self.n_gen = 0
        self.trend_best_network = []
        self.trend_time = []
        self.network_history = []
        self.score_history = []

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

    def mating(self, P):
        parents = select_parents(P)
        O = self.crossover.do(self.problem, parents, P, algorithm=self)
        O = self.mutation.do(self.problem, P, O, algorithm=self)
        return O

    def next(self, pop):
        offsprings = self.mating(pop)
        for network in offsprings:
            is_terminated = self.evaluate(network)
            if network.score > self.trend_best_network[-1].score:
                best_network = deepcopy(network)
            else:
                best_network = self.trend_best_network[-1]
            update_log(best_network=best_network, cur_network=network, algorithm=self)
            if is_terminated:
                return True

        pool = pop + offsprings
        self.pop = self.survival.do(pool, self.pop_size)
        return False

    def search(self, **kwargs):
        self.max_eval = kwargs['max_eval']
        self.max_time = kwargs['max_time']
        self.search_metric = kwargs['metric']

        assert self.pop_size is not None
        assert self.tournament_size is not None

        if self.warm_up:
            assert self.n_sample_warmup != 0
            assert self.metric_warmup is not None
        self._reset()

        best_network = None

        # Initialize population
        self.pop = self.initialize()
        for network in self.pop:
            is_terminated = self.evaluate(network)
            self.trend_time.append(self.total_time)

            if best_network is None:
                best_network = deepcopy(network)
            else:
                if network.score > best_network.score:
                    best_network = deepcopy(network)
            update_log(best_network=best_network, cur_network=network, algorithm=self)
            if is_terminated:
                break

        # After the population is seeded, proceed with evolving the population.
        while (self.n_eval <= self.max_eval) and (self.total_time <= self.max_time):
            self.n_gen += 1
            is_terminated = self.next(self.pop)
            if is_terminated:
                break
        return best_network


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


def compare(idv_1, idv_2):
    if idv_1.score > idv_2.score:
        return idv_1
    elif idv_1.score < idv_2.score:
        return idv_2
    else:
        return np.random.choice([idv_1, idv_2])


def select_parents(pop):
    parents_pool = []
    for _ in range(2):
        index_pool = np.random.permutation(len(pop)).reshape((len(pop) // 2, 2))
        for idx in index_pool:
            competitor_1, competitor_2 = pop[idx[0]], pop[idx[1]]
            winner = compare(competitor_1, competitor_2)
            parents_pool.append(winner)
    return np.array(parents_pool)