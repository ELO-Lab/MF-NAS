from algos import Algorithm
from models import Network
from copy import deepcopy
from algos.utils import sampling_solution, update_log

class RandomSearch(Algorithm):
    def __init__(self):
        super().__init__()
        self.trend_best_network = []
        self.trend_time = []
        self.network_history = []
        self.score_history = []

    def _reset(self):
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

    def search(self, **kwargs):
        self._reset()
        self.max_eval = kwargs['max_eval']
        self.max_time = kwargs['max_time']
        self.search_metric = kwargs['metric']

        best_network = Network()
        best_network.score = -99999999

        while True:
            network = sampling_solution(problem=self.problem)
            is_terminated = self.evaluate(network)

            if network.score > best_network.score:
                best_network = deepcopy(network)

            update_log(best_network=best_network, cur_network=network, algorithm=self)
            if is_terminated:
                return best_network
