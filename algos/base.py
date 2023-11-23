from abc import ABC, abstractmethod
from utils import set_seed

class Algorithm(ABC):
    def __init__(self):
        self.problem = None
        self.max_eval, self.max_time = None, None

        self.using_zc_metric = False
        self.metric = None
        self.iepoch = -1  # for training-based search objectives

        self.n_eval = 0
        self.total_time, self.total_epoch = 0.0, 0.0

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
        network, search_cost, total_epoch = self._run(**kwargs)
        return network, search_cost, total_epoch

    @abstractmethod
    def _run(self, **kwargs):
        pass