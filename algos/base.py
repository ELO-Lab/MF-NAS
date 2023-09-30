from abc import ABC, abstractmethod
from utils import set_seed

class Algorithm(ABC):
    def __init__(self):
        self.problem = None
        self.metric = None

    def adapt(self, problem, metric):
        self.problem = problem,
        self.metric = metric

    def run(self, seed, **kwargs):
        set_seed(seed)
        self._run(**kwargs)

    @abstractmethod
    def _run(self, **kwargs):
        pass