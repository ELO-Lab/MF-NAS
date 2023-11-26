from . import Algorithm
from models import Network
from copy import deepcopy
from .utils import sampling_solution, update_log

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
        self._reset()
        max_eval = self.problem.max_eval if self.max_eval is None else self.max_eval
        max_time = self.problem.max_time if self.max_time is None else self.max_time
        if not self.using_zc_metric and self.iepoch is None:
            raise ValueError

        best_network = self.search(max_eval=max_eval, max_time=max_time, metric=self.metric, iepoch=self.iepoch,
                                   **kwargs)
        return best_network, self.total_time, self.total_epoch

    def search(self, **kwargs):
        max_eval, max_time = kwargs['max_eval'], kwargs['max_time']
        metric, iepoch = kwargs['metric'], kwargs['iepoch']

        best_network = Network()
        best_network.score = -99999999

        while (self.n_eval < max_eval) and (self.total_time < max_time):
            network = sampling_solution(problem=self.problem)
            info, cost_time = self.evaluate(network, using_zc_metric=self.using_zc_metric, metric=metric, iepoch=iepoch)
            network.score = info[metric]

            self.total_time += cost_time
            self.total_epoch += self.iepoch

            if network.score > best_network.score:
                best_network = deepcopy(network)

            update_log(best_network=best_network, cur_network=network, algorithm=self)

        return best_network
