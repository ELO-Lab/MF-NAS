from abc import ABC, abstractmethod

class SearchSpace(ABC):
    def __init__(self):
        pass

    def sample(self, genotype=False):
        network = self._sample()
        if genotype:
            network = self.encode(network)
        return network

    def is_valid(self, network):
        return True

    @abstractmethod
    def return_available_ops(self, idx):
        pass

    @abstractmethod
    def _sample(self):
        pass

    @abstractmethod
    def encode(self, network):
        pass

    @abstractmethod
    def decode(self, network):
        pass