from abc import ABC, abstractmethod
from utils import set_seed

class Algorithm(ABC):
    def __init__(self):
        self.problem = None
        self.max_eval, self.max_time = None, None

        self.using_zc_metric = False
        self.metric = None
        self.iepoch = 0  # for training-based search objectives

        self.n_eval = 0
        self.total_time, self.total_epoch = 0.0, 0.0

        self.search_log = []

    def evaluate(self, network, using_zc_metric, metric):
        cost_time = self.problem.evaluate(network, using_zc_metric=using_zc_metric, metric=metric)
        self.n_eval += 1
        return cost_time

    def reset(self):
        self.n_eval = 0
        self.total_time, self.total_epoch = 0.0, 0.0

    def set(self, configs):
        for key, value in configs.items():
            setattr(self, key, value)

    def adapt(self, problem):
        self.problem = problem

    def run(self, seed, **kwargs):
        set_seed(seed)
        self.reset()
        best_network, search_cost, total_epoch = self._run(**kwargs)

        return best_network, search_cost, total_epoch

    @abstractmethod
    def _run(self, **kwargs):
        pass