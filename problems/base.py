from abc import ABC, abstractmethod

class Problem(ABC):
    def __init__(self, search_space, max_eval, max_time):
        self.search_space = search_space
        self.max_eval, self.max_time = max_eval, max_time

    @abstractmethod
    def evaluate(self, network, metric, **kwargs):
        pass