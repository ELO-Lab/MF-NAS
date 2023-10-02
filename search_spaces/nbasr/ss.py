from search_spaces import SearchSpace
import numpy as np

main_ops = [0, 1, 2, 3, 4, 5]
idx_main_ops = [0, 2, 5]

skip_ops = [0, 1]
idx_skip_ops = [1, 3, 4, 6, 7, 8]
max_length = 9

class SS_ASR(SearchSpace):
    def __init__(self):
        super().__init__()

    def _sample(self):
        network = np.zeros(max_length, dtype=np.int8)
        network[idx_main_ops] = np.random.choice(main_ops, len(idx_main_ops))
        network[idx_skip_ops] = np.random.choice(skip_ops, len(idx_skip_ops))
        network = self.decode(network)
        return network

    def encode(self, network):
        encoded_network = np.zeros(max_length, dtype=np.int8)
        encoded_network[0], encoded_network[1] = network[0][0], network[0][1]
        encoded_network[2], encoded_network[3], encoded_network[4] = network[1][0], network[1][1], network[1][2]
        encoded_network[5], encoded_network[6] = network[2][0], network[2][1]
        encoded_network[7], encoded_network[8] = network[2][2], network[2][3]
        return encoded_network

    def decode(self, encode_network):
        network = [(encode_network[0], encode_network[1]), (encode_network[2], encode_network[3], encode_network[4]),
                    (encode_network[5], encode_network[6], encode_network[7], encode_network[8])]
        return network

    def return_available_ops(self, idx):
        return main_ops if idx in idx_main_ops else skip_ops  # encoded_ops


if __name__ == '__main__':
    # np.random.seed(0)
    # ss = SS_ASR()
    # network = ss.sample()
    # encode_network = ss.encode(network)
    # decode_network = ss.decode(encode_network)
    # print(network, encode_network, decode_network)
    pass