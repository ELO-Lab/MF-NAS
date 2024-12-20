from torch.utils.benchmark.utils.fuzzer import dtype_size

from search_spaces import SearchSpace
from search_spaces.nb101.utils import OutOfDomainError, check_spec, ModelSpec_
from search_spaces.nb101.model import Network
import numpy as np

allowed_ops = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
allowed_edges = [0, 1]  # Binary adjacency matrix
num_vertices = 7
edge_spots_idx = np.triu_indices(num_vertices, 1)
edge_spots = int(num_vertices * (num_vertices - 1) / 2)  # Upper triangular matrix
op_spots = int(num_vertices - 2)  # Input/output vertices are fixed

class SS_101(SearchSpace):
    def __init__(self):
        super().__init__('NB-101')

    def _sample(self):
        matrix = np.random.choice(allowed_edges, size=(num_vertices, num_vertices))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(allowed_ops, size=num_vertices).tolist()
        ops[0] = 'input'
        ops[-1] = 'output'
        return {'matrix': matrix, 'ops': ops}

    def encode(self, network: dict):
        x_edge = np.array(network['matrix'])[edge_spots_idx]
        x_ops = np.empty(num_vertices - 2)
        for i, op in enumerate(network['ops'][1:-1]):
            x_ops[i] = (np.array(allowed_ops) == op).nonzero()[0][0]
        encode_network = np.concatenate((x_edge, x_ops))
        return [int(v) for v in encode_network]

    def decode(self, encode_network: np.ndarray):
        network_edge = encode_network[:edge_spots]
        network_ops = encode_network[-op_spots:]
        matrix = np.zeros((num_vertices, num_vertices), dtype=int)
        matrix[edge_spots_idx] = network_edge
        ops = ['input'] + [allowed_ops[i] for i in network_ops] + ['output']
        return {'matrix': matrix, 'ops': ops}

    def return_available_ops(self, idx):
        return [0, 1] if idx in range(21) else [0, 1, 2]  # encoded_ops

    def is_valid(self, encode_network: np.ndarray):
        network = self.decode(encode_network)
        model_spec = ModelSpec_(network['matrix'], network['ops'])
        try:
            check_spec(model_spec)
        except OutOfDomainError:
            return False
        return True

    def get_model(self, genotype, num_classes=10):
        phenotype = self.decode(genotype)
        model_spec = ModelSpec_(phenotype['matrix'], phenotype['ops'])
        network = Network(model_spec, stem_out=128, num_stacks=3, num_mods=3, num_classes=num_classes)
        return network

if __name__ == '__main__':
    # np.random.seed(0)
    # ss = SS_101()
    # phenotype = ss.sample()
    # genotype = ss.encode(phenotype)
    # network = ss.get_model(genotype)
    # print('Phenotype:', phenotype)
    # print('Genotype:', genotype)
    # print('Network:', network)
    pass