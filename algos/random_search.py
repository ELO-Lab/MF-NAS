from . import Algorithm
from models import Network
from copy import deepcopy

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
        if not self.using_zc_metric:
            metric = self.metric + f'_{self.iepoch}'
        else:
            metric = self.metric
        n_eval = 0
        total_time = 0

        network = Network()
        network.genotype = self.problem.search_space.sample(genotype=True)
        time = self.problem.evaluate(network, using_zc_metric=self.using_zc_metric, metric=metric)

        n_eval += 1
        total_time += time

        best_network = deepcopy(network)

        self.trend_best_network = [best_network]
        self.trend_time = [total_time]

        self.network_history, self.score_history = [deepcopy(network)], [best_network.score]
        while (n_eval <= self.problem.max_eval) and (total_time <= self.problem.max_time):
            network = Network()
            network.genotype = self.problem.search_space.sample(genotype=True)
            time = self.problem.evaluate(network, using_zc_metric=self.using_zc_metric, metric=metric)

            if network.score > best_network.score:
                best_network = deepcopy(network)
            self.trend_best_network.append(best_network)
            self.network_history.append(deepcopy(network))
            self.score_history.append(network.score)

            n_eval += 1
            total_time += time

        best_network = self.trend_best_network[-1]
        search_time = total_time
        return best_network, search_time
