from . import Algorithm
from models import Network
import numpy as np
from algos.rea import run_warm_up
from .utils import sampling_solution, update_log
from copy import deepcopy

class MFGA(Algorithm):
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
        self.init_iepoch = 4
        self.step = 2

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

    def initialize(self):
        if not self.warm_up:
            pop = []
            for _ in range(self.pop_size):
                network = sampling_solution(problem=self.problem)
                pop.append(network)
        else:
            pop, warmup_time = run_warm_up(self.n_sample_warmup, self.pop_size, self.problem, self.metric_warmup)
        return pop
    # def set_age(self):
    #     for network in self.pop:
    #         network.info[]

    def search(self, **kwargs):
        max_eval, max_time = kwargs['max_eval'], kwargs['max_time']
        metric, iepoch = kwargs['metric'], kwargs['iepoch']

        # best_network = Network()
        # best_network.score = -np.inf

        # Initialize pop
        self.pop = self.initialize()

        # Evaluate initial pop
        for network in self.pop:
            iepoch = self.init_iepoch
            info, cost_time = self.evaluate(network, using_zc_metric=self.using_zc_metric, metric=metric, iepoch=iepoch)
            network.score = info[metric]
            network.info['train_info'][iepoch]['score'] = network.score

            diff_epoch = network.info['cur_iepoch'][-1] - network.info['cur_iepoch'][-2]
            self.total_time += cost_time
            self.total_epoch += diff_epoch

            self.trend_time.append(self.total_time)

        print('init ----')
        for sol in self.pop:
            print(sol.genotype, sol.info['train_info'])

        while (self.n_eval < max_eval) and (self.total_time < max_time):
            parents = selection(self.pop, tournament_size=2, n_survive=self.pop_size)

            ## Crossover
            offsprings = crossover(parents=parents, n_offspring=self.pop_size,
                                   prob_crossover=self.prob_crossover, crossover_method=self.crossover_method,
                                   problem=self.problem)
            print('crossover ----')
            for sol in offsprings:
                print(sol.info)

            ## Mutate
            offsprings = mutation(pool=offsprings, prob_mutation=self.prob_mutation, problem=self.problem)
            print('mutate ----')
            for sol in offsprings:
                print(sol.info)

            ## Evaluate offsprings
            for network in offsprings:
                if network.info['cur_iepoch'][-1] == 0:
                    iepoch = self.init_iepoch
                    info, cost_time = self.evaluate(network, using_zc_metric=self.using_zc_metric, metric=metric, iepoch=iepoch)
                    network.score = info[metric]
                    network.info['train_info'][iepoch]['score'] = network.score
                    diff_epoch = network.info['cur_iepoch'][-1] - network.info['cur_iepoch'][-2]

                    self.total_time += cost_time
                    self.total_epoch += diff_epoch

                    self.trend_time.append(self.total_time)

            ## Selection
            pool = parents + offsprings
            print(len(pool))
            for sol in pool:
                print(sol.genotype, sol.info['train_info'])
            self.pop = selection(pool=pool, tournament_size=self.tournament_size, n_survive=self.pop_size)

            ## Reward for winners -> Evaluate more epochs
            for network in self.pop:
                iepoch = min(network.info['cur_iepoch'][-1] + self.step, 200)
                info, cost_time = self.evaluate(network, using_zc_metric=self.using_zc_metric, metric=metric,
                                                iepoch=iepoch)
                network.score = info[metric]
                network.info['train_info'][iepoch]['score'] = network.score
                diff_epoch = network.info['cur_iepoch'][-1] - network.info['cur_iepoch'][-2]

                self.total_time += cost_time
                self.total_epoch += diff_epoch

                self.trend_time.append(self.total_time)

            print('selection ----')
            print(len(self.pop))
            for sol in self.pop:
                print(sol.genotype, sol.info['train_info'])
            print()

        # best_network = selection(pool=self.pop, tournament_size=len(self.pop), n_survive=1)[-1]
        # print(best_network.info)
        max_iepoch = max([candidate.info['cur_iepoch'][-1] for candidate in self.pop])
        for network in self.pop:
            if network.info['cur_iepoch'][-1] != max_iepoch:
                info, cost_time = self.evaluate(network, using_zc_metric=self.using_zc_metric, metric=metric, iepoch=max_iepoch)
                network.score = info[metric]
                network.info['train_info'][max_iepoch]['score'] = network.score
                diff_epoch = network.info['cur_iepoch'][-1] - network.info['cur_iepoch'][-2]

                self.total_time += cost_time
                self.total_epoch += diff_epoch

                self.trend_time.append(self.total_time)
        list_fitness = [candidate.info['train_info'][max_iepoch]['score'] for candidate in self.pop]
        best_network = self.pop[np.argmax(list_fitness)]

        for sol in self.pop:
            print(sol.genotype, sol.info['train_info'])
        return best_network

def mutation(pool, prob_mutation, problem):
    op_mutation_prob = prob_mutation / len(pool[0].genotype)

    mutated_pool = []
    for network in pool:
        if np.random.random() < 0.5:
            while True:
                new_genotype = network.genotype.copy()
                for i in range(len(new_genotype)):
                    if np.random.random() < op_mutation_prob:
                        available_ops = problem.search_space.return_available_ops(i).copy()
                        _available_ops = [o for o in available_ops if o != new_genotype[i]]
                        new_genotype[i] = np.random.choice(_available_ops)
                if problem.search_space.is_valid(new_genotype):
                    new_network = Network()
                    new_network.genotype = new_genotype
                    break
        else:
            new_network = deepcopy(network)
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
                offspring_net1, offspring_net2 = deepcopy(pair[0]), deepcopy(pair[1])
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
    # for candidate in pool:
    #     print(candidate.info)
    n_epoch = [len(candidate.info['cur_iepoch']) for candidate in pool]
    pos = {}
    for i, epoch in enumerate(n_epoch):
        if epoch not in pos:
            pos[epoch] = [i]
        else:
            pos[epoch].append(i)
    list_epoch = list(pos.keys())
    list_epoch.sort(reverse=True)

    while len(pool_survive) < n_survive:
        for epoch in list_epoch:
            i = np.array(pos[epoch])
            _pool = pool[i]
            np.random.shuffle(_pool)
            # if len(_pool) < tournament_size:
            #     pool_survive += _pool.tolist()
            # else:
            _I = []
            list_index = list(range(len(_pool)))
            np.random.shuffle(list_index)
            for j in range(0, len(list_index), tournament_size):
                _I.append(list_index[j:j+tournament_size])
            # print(len(_pool), _I, j)
            # if len(list_index) % tournament_size != 0:
            #     _I.append([list_index[j:]])
            # print(_I, len(_I))
            # _I = np.random.permutation(len(_pool)).reshape((len(_pool) // tournament_size, tournament_size))
            for _i in _I:
                list_candidates = _pool[np.array(_i)]
                # for candidate in list_candidates:
                #     print(candidate.info)

                # New
                # print([candidate.info['cur_iepoch'][-1] for candidate in list_candidates])
                min_iepoch = min([candidate.info['cur_iepoch'][-1] for candidate in list_candidates])
                list_fitness = [candidate.info['train_info'][min_iepoch]['score'] for candidate in list_candidates]
                winner = list_candidates[np.argmax(list_fitness)]
                pool_survive.append(winner)
    return pool_survive[:n_survive]

    # while len(pool_survive) < n_survive:
    #     np.random.shuffle(pool)
    #     for _ in range(tournament_size):
    #         I = np.random.permutation(len(pool)).reshape((len(pool) // tournament_size, tournament_size))
    #         for i in I:
    #             list_candidates = pool[i]
    #             # for candidate in list_candidates:
    #             #     print(candidate.info)
    #             # New
    #             min_iepoch = min([candidate.info['cur_iepoch'][-1] for candidate in list_candidates])
    #             list_fitness = [candidate.info['train_info'][min_iepoch]['score'] for candidate in list_candidates]
    #             winner = list_candidates[np.argmax(list_fitness)]
    #             pool_survive.append(winner)
    # return pool_survive[:n_survive]

