from algos import Algorithm
from models import Network
from copy import deepcopy
import numpy as np
import math
from algos.utils import sampling_solution

class SuccessiveHalving(Algorithm):
    def __init__(self):
        super().__init__()
        self.list_iepoch = None
        self.n_candidate = -1

    def _run(self):
        assert len(self.list_iepoch) is not None
        max_time = self.problem.max_time if self.max_time is None else self.max_time

        list_network = self.sample()
        best_network = self.search(list_network, max_time=max_time)
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
        self.total_time += total_time
        self.total_epoch += total_epoch
        return is_terminated

    def sample(self):
        list_network = []
        for _ in range(self.n_candidate):
            network = sampling_solution(self.problem)
            list_network.append(network)
        return list_network

    def search(self, list_network, **kwargs):
        assert len(list_network) != 0
        self.max_time = kwargs['max_time']
        list_network = np.array(list_network)

        best_network = Network()
        if 'loss' in self.metric or 'per' in self.metric:
            best_network.score *= -1

        for iepoch in self.list_iepoch:
            evaluated_network = []
            network_scores = []

            is_terminated = self.evaluate(list_network, using_zc_metric=False, metric=self.metric + f'_{iepoch}')

            for network in list_network:
                evaluated_network.append(network)
                network_scores.append(network.score)

                if network.score > best_network.score:
                    best_network = deepcopy(network)

            if is_terminated:
                return best_network

            # ids = np.flip(np.argsort(network_scores))
            # list_network = np.array(evaluated_network)[ids]
            # list_network = list_network[:math.ceil(len(list_network) / 2)]
            # if len(list_network) == 1 or iepoch == self.list_iepoch[-1]:
            #     return best_network

            if len(list_network) == 1 or iepoch == self.list_iepoch[-1]:
                list_test_acc = [self.problem.get_test_performance(network)[0] for network in list_network]
                # return best_network
                return list_network[np.argmax(list_test_acc)]

            ids = np.flip(np.argsort(network_scores))
            list_network = np.array(evaluated_network)[ids]
            list_network = list_network[:math.ceil(len(list_network) / 2)]

