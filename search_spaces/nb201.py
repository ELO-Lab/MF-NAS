from search_spaces import SearchSpace
import numpy as np

list_ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

def get_op(x):
    return x.split('~')[0]

class SS_201(SearchSpace):
    def __init__(self):
        super().__init__()
        self.available_ops = list(range(len(list_ops)))  # encoded_ops

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

if __name__ == '__main__':
    # np.random.seed(0)
    # ss = NB_201()
    # network = ss.sample()
    # encode_network = ss.encode(network)
    # decode_network = ss.decode(encode_network)
    # print(network, encode_network, decode_network)
    pass