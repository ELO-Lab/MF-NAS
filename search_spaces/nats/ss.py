import numpy as np
from utils import dict2config
from search_spaces import SearchSpace
from search_spaces.nats.model import Structure as CellStructure, DynamicShapeTinyNet

class SS_NATS(SearchSpace):
    def __init__(self):
        super().__init__('NATS')

        self.n_var = 5
        self.lb = [0] * self.n_var
        self.ub = [7] * self.n_var
        self.allowed_channels = ['8', '16', '24', '32', '40', '48', '56', '64']

        self.categories = [list(range(a, b + 1)) for a, b in zip(self.lb, self.ub)]

    def _sample(self):
        sampled_ops = np.random.choice(self.allowed_channels, self.n_var)
        network = '{}:{}:{}:{}:{}'.format(*sampled_ops)
        return network

    def encode(self, network):
        nodes = network.split(':')
        encode_network = [self.allowed_channels.index(channel) for channel in nodes]
        return encode_network

    def decode(self, encode_network):
        channels = [self.allowed_channels[idx] for idx in encode_network]
        network = '{}:{}:{}:{}:{}'.format(*channels)
        return network

    def return_available_ops(self, idx, **kwargs):
        return self.categories[idx]

    def get_model(self, genotype, num_classes=10):
        channels = self.decode(genotype)
        config = {
            'name': 'infer.shape.tiny',
            'channels': channels,
            'genotype': '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|',
            'num_classes': num_classes
        }
        if isinstance(config, dict):
            config = dict2config(config, None)
        if isinstance(config.channels, str):
            channels = tuple([int(x) for x in config.channels.split(":")])
        else:
            channels = config.channels
        genotype = CellStructure.str2structure(config.genotype)
        network = DynamicShapeTinyNet(channels, genotype, config.num_classes)
        return network

if __name__ == '__main__':
    # np.random.seed(0)
    # ss = SS_NATS()
    # network = ss.sample()
    # genotype = ss.encode(network)
    # print(genotype)
    # print(network)
    # genotype = ss.encode(network)
    # print(genotype)
    # print(ss.decode(genotype))
    # print(ss.return_available_ops(0))
    # print(ss.get_model(genotype=genotype, num_classes=10))
    pass