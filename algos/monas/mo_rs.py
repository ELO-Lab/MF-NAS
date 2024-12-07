from algos import Algorithm
from algos.utils import sampling_solution, ElitistArchive


class MultiObjective_RandomSearch(Algorithm):
    def __init__(self):
        super().__init__(nas_type='mo')
        self.trend_best_network = []
        self.trend_time = []
        self.network_history = []
        self.archive = ElitistArchive()

        self.list_metrics, self.list_iepochs, self.need_trained = [], [], []
        self.visited = []
        self.explored_networks = []

    def _reset(self):
        self.trend_best_network = []
        self.trend_time = []
        self.network_history = []
        self.archive = ElitistArchive()
        self.visited = []
        self.explored_networks = []

    def reformat_list_metrics(self):
        list_metrics = []
        for i in range(len(self.need_trained)):
            if self.need_trained[i]:
                list_metrics.append(self.list_metrics[i] + f'_{self.list_iepochs[i]}')
            else:
                list_metrics.append(self.list_metrics[i])
        return list_metrics

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
        max_eval = self.problem.max_eval if self.max_eval is None else self.max_eval
        max_time = self.problem.max_time if self.max_time is None else self.max_time
        approximation_set = self.search(max_eval=max_eval, max_time=max_time, **kwargs)
        return approximation_set, self.total_time, self.total_epoch

    def search(self, **kwargs):
        max_eval = kwargs['max_eval']
        max_time = kwargs['max_time']
        list_metrics = self.reformat_list_metrics()

        while (self.n_eval <= max_eval) and (self.total_time <= max_time):
            while True:
                network = sampling_solution(self.problem)
                _hash = self.problem.get_hash(network)
                if _hash not in self.visited:
                    self.visited.append(_hash)
                    break
            train_time, train_epoch, is_terminated = self.problem.mo_evaluate(list_networks=[network],
                                                                              list_metrics=list_metrics,
                                                                              need_trained=self.need_trained,
                                                                              cur_total_time=self.total_time,
                                                                              max_time=max_time)
            self.network_history.append(network)
            self.archive.update(network, problem=self.problem)
            self.n_eval += 1
            self.total_time += train_time
            self.total_epoch += train_epoch
            self.explored_networks.append([self.total_time, network.genotype, self.problem.get_hash(network), network.score])
            if is_terminated or self.n_eval >= max_eval:
                return self.archive
