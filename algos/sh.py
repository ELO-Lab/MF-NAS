from . import Algorithm
from models import Network
from copy import deepcopy
import numpy as np
import math
from .utils import sampling_solution

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

    def sample(self):
        list_network = []
        for _ in range(self.n_candidate):
            network = sampling_solution(self.problem)
            list_network.append(network)
        return list_network

    def search(self, list_network, **kwargs):
        assert len(list_network) != 0
        max_time = kwargs['max_time']
        checkpoint = 0
        iepoch = self.list_iepoch[checkpoint]

        list_network = np.array(list_network)

        last_iepoch = 0
        best_network = Network()
        best_network.score = -np.inf

        last = False
        while self.total_time <= max_time:
            evaluated_network = []
            network_scores = []

            for network in list_network:
                time = self.problem.evaluate(network, using_zc_metric=self.using_zc_metric, metric=self.metric+f'_{iepoch}')
                diff_epoch = network.info['cur_iepoch'][-1] - last_iepoch
                self.total_time += time
                self.total_epoch += diff_epoch

                evaluated_network.append(network)
                network_scores.append(network.score)

                if self.total_time >= max_time:
                    self.total_time -= time
                    self.total_epoch -= diff_epoch

                    return best_network

                if network.score > best_network.score:
                    best_network = deepcopy(network)

            ids = np.flip(np.argsort(network_scores))
            list_network = np.array(evaluated_network)[ids]
            list_network = list_network[:math.ceil(len(list_network) / 2)]
            if len(list_network) == 1 or last:
                return best_network

            checkpoint += 1
            last_iepoch = iepoch
            iepoch = self.list_iepoch[checkpoint]
            if iepoch == self.list_iepoch[-1]:
                last = True
