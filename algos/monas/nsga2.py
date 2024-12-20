import numpy as np

from algos.base import Algorithm
from algos.utils import sampling_solution, ElitistArchive
from utils.genetic_operators import PointCrossover, BitStringMutation, RankAndCrowdingSurvival


class NSGA2(Algorithm):
    def __init__(self):
        super().__init__(nas_type='mo')
        self.pop = []
        self.pop_size = 0
        self.n_gen = 0
        self.archive = ElitistArchive()

        self.prob_c, self.prob_m = -1, -1
        self.crossover_method = None
        self.crossover, self.mutation, self.survival = None, None, None

        self.list_metrics, self.list_iepochs, self.need_trained = [], [], []
        self.explored_networks = []

    def get_genetic_operators(self):
        self.crossover = PointCrossover(prob=self.prob_c, method=self.crossover_method)
        self.mutation = BitStringMutation(prob=self.prob_m)
        self.survival = RankAndCrowdingSurvival()

    def _reset(self):
        self.pop = []
        self.n_gen = 0
        self.archive = ElitistArchive()
        self.explored_networks = []

    def finalize(self, **kwargs):
        try:
            save_path = kwargs['save_path']
            rid = kwargs['rid']
            import pickle as p
            p.dump(self.explored_networks, open(save_path + f'/explored_networks_run{rid}.p', 'wb'))
        except KeyError:
            pass

    def _run(self, **kwargs):
        self._reset()
        self.get_genetic_operators()

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
        max_eval = kwargs['max_eval']
        max_time = kwargs['max_time']
        list_metrics = kwargs['list_metrics']
        is_terminated = self._initialize(list_metrics=list_metrics, max_time=max_time)
        if is_terminated:
            return self.archive

        self.pop = self.survival.do(self.pop, self.pop_size)
        while (self.n_eval <= max_eval) and (self.total_time <= max_time):
            self.n_gen += 1
            is_terminated = self._next(self.pop, list_metrics=list_metrics, max_time=max_time, max_eval=max_eval)
            if is_terminated:
                break
        return self.archive

    def _initialize(self, **kwargs):
        list_metrics, max_time = kwargs['list_metrics'], kwargs['max_time']
        pop_hashes = []
        n = 0
        while True:
            network = sampling_solution(self.problem)
            network_hash = self.problem.get_hash(network)
            if network_hash not in pop_hashes:
                pop_hashes.append(network_hash)
                train_time, train_epoch, is_terminated = self.problem.mo_evaluate(list_networks=[network],
                                                                                  list_metrics=list_metrics,
                                                                                  need_trained=self.need_trained,
                                                                                  cur_total_time=self.total_time,
                                                                                  max_time=max_time)
                self.archive.update(network, problem=self.problem)
                self.n_eval += 1
                self.total_time += train_time
                self.total_epoch += train_epoch
                self.pop.append(network)
                self.explored_networks.append([self.total_time, network.genotype, network_hash, network.score.copy()])
                n += 1
                if n == self.pop_size or is_terminated:
                    return is_terminated

    def _mating(self, P, **kwargs):
        # Selection
        parents = select_parents(P)

        # Crossover
        O = self.crossover.do(self.problem, parents, P, algorithm=self)

        # Mutation
        O = self.mutation.do(self.problem, P, O, algorithm=self)

        return O

    def _next(self, pop, **kwargs):
        """
         Workflow in 'Next' step:
        + Create the offspring.
        + Select the new population.
        """
        list_metrics, max_time, max_eval = kwargs['list_metrics'], kwargs['max_time'], kwargs['max_eval']

        offsprings = self._mating(pop)
        for network in offsprings:
            train_time, train_epoch, is_terminated = self.problem.mo_evaluate(list_networks=[network],
                                                                              list_metrics=list_metrics,
                                                                              need_trained=self.need_trained,
                                                                              cur_total_time=self.total_time,
                                                                              max_time=max_time)
            self.archive.update(network, problem=self.problem)
            self.n_eval += 1
            self.total_time += train_time
            self.total_epoch += train_epoch
            self.explored_networks.append([self.total_time, network.genotype, self.problem.get_hash(network), network.score.copy()])

            if is_terminated or self.n_eval >= max_eval:
                return True

        pool = pop + offsprings
        self.pop = self.survival.do(pool, self.pop_size)
        return False

def compare(idv_1, idv_2):
    rank_1, rank_2 = idv_1.get('rank'), idv_2.get('rank')
    if rank_1 < rank_2:
        return idv_1
    elif rank_1 > rank_2:
        return idv_2
    else:
        cd_1, cd_2 = idv_1.get('crowding'), idv_2.get('crowding')
        if cd_1 > cd_2:
            return idv_1
        elif cd_1 < cd_2:
            return idv_2
        else:
            return idv_1

def select_parents(pop):
    parents_pool = []
    for _ in range(2):
        index_pool = np.random.permutation(len(pop)).reshape((len(pop) // 2, 2))
        for idx in index_pool:
            competitor_1, competitor_2 = pop[idx[0]], pop[idx[1]]
            winner = compare(competitor_1, competitor_2)
            parents_pool.append(winner)
    return np.array(parents_pool)