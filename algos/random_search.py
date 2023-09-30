from . import Algorithm
from models import Network
from copy import deepcopy

class RandomSearch(Algorithm):
    def __init__(self):
        super().__init__()

    def _run(self, **kwargs):
        n_eval = 0
        total_time = 0

        network = Network()
        network.genotype = self.problem.search_space.sample(genotype=True)
        time = self.problem.evaluate(network, algorithm=self)

        n_eval += 1
        total_time += time

        best_network = deepcopy(network)

        self.trend_best_network = [best_network]
        self.trend_time = [total_time]

        self.network_history, self.score_history = [deepcopy(network)], [best_network.score]
        while (n_eval <= self.problem.max_eval) and (total_time <= self.problem.max_time):
            network = Network()
            network.genotype = self.problem.search_space.sample(genotype=True)
            time = self.problem.evaluate(network, algorithm=self)

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
