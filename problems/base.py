from abc import ABC, abstractmethod

class Problem(ABC):
    def __init__(self, search_space, max_eval, max_time):
        self.search_space = search_space
        self.max_eval, self.max_time = max_eval, max_time

    def evaluate(self, network, **kwargs):
        using_zc_metric = bool(kwargs['using_zc_metric'])
        if using_zc_metric:
            metric = kwargs['metric']
            info, time = self.zc_evaluate(network, metric=metric)
        else:
            iepoch = kwargs['iepoch']
            info, time = self.train(network, iepoch=iepoch)
        return info, time

    @abstractmethod
    def get_test_performance(self, network, **kwargs):
        pass

    @abstractmethod
    def zc_evaluate(self, network, **kwargs):
        pass

    @abstractmethod
    def train(self, network, **kwargs):
        pass