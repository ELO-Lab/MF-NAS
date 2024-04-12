from search_spaces import SearchSpace
from collections import namedtuple
from search_spaces.ptb_darts.model import RNNModel
import numpy as np

Genotype = namedtuple('Genotype', 'recurrent concat')

PRIMITIVES = [
    'none',
    'tanh',
    'relu',
    'sigmoid',
    'identity'
]
STEPS = 8
CONCAT = 8

class SS_PTB_DARTS(SearchSpace):
    def __init__(self):
        super().__init__('DARTS')

        self.n_var = 16
        self.n_ops = 5
        self.lb = [0] * self.n_var
        self.ub = [1] * self.n_var
        self.allowed_ops = PRIMITIVES

        h = 1
        for b in range(0, self.n_var // 1, 4):
            self.ub[b] = self.n_ops - 1
            self.ub[b + 1] = h
            self.ub[b + 2] = self.n_ops - 1
            self.ub[b + 3] = h
            h += 1
        self.ub[self.n_var // 1:] = self.ub[:self.n_var // 1]
        self.categories = [list(range(a, b + 1)) for a, b in zip(self.lb, self.ub)]

    def _sample(self):
        x = []
        for i in range(1, self.n_var, 4):
            inp = np.random.choice(self.categories[i], 2, replace=False)
            x.extend([np.random.choice(self.n_ops), inp[0],
                      np.random.choice(self.n_ops), inp[1]])
        return self.decode(x)

    def encode(self, network):
        # Phenotype -> Genotype (an integer vector)
        x = []
        for unit in network.recurrent:
            x.append((np.array(self.allowed_ops) == unit[0]).nonzero()[0][0])
            x.append(unit[1])
        return np.array(x)

    def decode(self, encode_network):
        genome = self._convert(encode_network)

        # decodes genome to architecture
        recurrent_cell = self._decode_cell(genome)
        return Genotype(recurrent=recurrent_cell, concat=range(1, 9))

    def return_available_ops(self, idx, **kwargs):
        return self.categories[idx]

    def is_valid(self, encode_network: np.ndarray):
        for i in range(0, 16, 4):
            if encode_network[i+1] == encode_network[i+3]:
                return False
        return True

    def get_model(self, genotype, search=True):
        phenotype = self.decode(genotype)
        if search:
            return RNNModel(ntoken=10000, ninp=300, nhid=300, nhidlast=300,
                            dropout=0.75, dropouth=0.25, dropoutx=0.75, dropouti=0.2, dropoute=0, genotype=phenotype)
        return RNNModel(ntoken=10000, ninp=850, nhid=850, nhidlast=850,
                        dropout=0.75, dropouth=0.25, dropoutx=0.75, dropouti=0.2, dropoute=0.1, genotype=phenotype)

    # ------------------------ Following functions are specific to DARTS search space ------------------------- #
    @staticmethod
    def _convert_cell(cell_bit_string):
        # convert cell bit-string to genome
        tmp = [cell_bit_string[i:i + 2] for i in range(0, len(cell_bit_string), 2)]
        return [tmp[i:i + 2] for i in range(0, len(tmp), 2)]

    def _convert(self, bit_string):
        # convert network bit-string (norm_cell + reduce_cell) to genome
        gene = self._convert_cell(bit_string[:len(bit_string)])
        return gene

    def _decode_cell(self, cell_genome):
        cell = []
        for block in cell_genome:
            for unit in block:
                cell.append((self.allowed_ops[unit[0]], unit[1]))
                # the following lines are for NASNet search space, DARTS simply concat all nodes outputs
                # if unit[1] in cell_concat:
                #     cell_concat.remove(unit[1])
        return cell

if __name__ == '__main__':
    np.random.seed(0)
    ss = SS_PTB_DARTS()
    genotype = ss.sample(True)
    model = ss.get_model(genotype, True)
    print(model)
    size = 0
    for p in model.parameters():
        size += p.nelement()
    print('param size: {}'.format(size))
    print('initial genotype:')
    print(model.rnns[0].genotype)
    # print(network)
    # genotype = ss.encode(network)
    # for i in range(16):
    #     print(ss.return_available_ops(i))
    pass