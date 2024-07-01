from search_spaces import SearchSpace
from search_spaces.nb201.model import TinyNetwork
from search_spaces.nb201.utils import dict2config, Structure as CellStructure
import numpy as np

list_ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']


def get_op(x):
    return x.split('~')[0]


class SS_201(SearchSpace):
    def __init__(self):
        super().__init__('NB-201')

    def _sample(self):
        sampled_ops = np.random.choice(list_ops, 6)
        network = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*sampled_ops)
        return network

    def encode(self, network):
        nodes = network.split('+')
        node_ops = [list(map(get_op, n.strip()[1:-1].split('|'))) for n in nodes]
        encode_network = [list_ops.index(op) for ops in node_ops for op in ops]
        return encode_network

    def decode(self, encode_network):
        ops = [list_ops[idx] for idx in encode_network]
        network = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*ops)
        return network

    def return_available_ops(self, idx):
        return list(range(len(list_ops)))  # encoded_ops

    def get_model(self, genotype, num_classes=10):
        phenotype = self.decode(genotype)
        config = {
            'name': 'infer.tiny',
            'C': 16,
            'N': 5,
            'arch_str': phenotype,
            'num_classes': num_classes
        }
        if isinstance(config, dict):
            config = dict2config(config, None)
        genotype = CellStructure.str2structure(config.arch_str)
        network = TinyNetwork(C=config.C, N=config.N, genotype=genotype, num_classes=config.num_classes)
        return network

if __name__ == '__main__':
    # np.random.seed(0)
    # ss = SS_201()
    # phenotype = ss.sample()
    # genotype = ss.encode(phenotype)
    # network = ss.get_model(genotype)
    # print('Phenotype:', phenotype)
    # print('Genotype:', genotype)
    # print('Network:', network)
    pass