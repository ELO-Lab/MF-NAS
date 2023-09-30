from abc import ABC, abstractmethod
from utils import set_seed

class Algorithm(ABC):
    def __init__(self):
        self.problem = None
        self.metric = None

    def set(self, configs):
        for key, value in configs.items():
            setattr(self, key, value)

    def adapt(self, problem):
        self.problem = problem

    def run(self, seed, **kwargs):
        set_seed(seed)
        network, search_cost = self._run(**kwargs)
        return network, search_cost

    @abstractmethod
    def _run(self, **kwargs):
        pass