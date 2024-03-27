"""
Inspired by: EvoxBench
"""
from search_spaces import SearchSpace
from collections import namedtuple
from search_spaces.darts.model import NetworkCIFAR
import numpy as np

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")
Genotype_cell = namedtuple('CellGenotype', 'cell concat')
PRIMITIVES = [
    'skip_connect',
    'avg_pool_3x3',
    'max_pool_3x3',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]

class SS_DARTS(SearchSpace):
    def __init__(self):
        super().__init__('DARTS')

        self.n_var = 32
        self.n_ops = 7
        self.lb = [0] * self.n_var
        self.ub = [1] * self.n_var
        self.allowed_ops = PRIMITIVES

        h = 1
        for b in range(0, self.n_var // 2, 4):
            self.ub[b] = self.n_ops - 1
            self.ub[b + 1] = h
            self.ub[b + 2] = self.n_ops - 1
            self.ub[b + 3] = h
            h += 1
        self.ub[self.n_var // 2:] = self.ub[:self.n_var // 2]
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
        # normal cell
        for unit in network.normal:
            x.append((np.array(self.allowed_ops) == unit[0]).nonzero()[0][0])
            x.append(unit[1])

        # reduction cell
        for unit in network.reduce:
            x.append((np.array(self.allowed_ops) == unit[0]).nonzero()[0][0])
            x.append(unit[1])

        return np.array(x)

    def decode(self, encode_network):
        genome = self._convert(encode_network)
        # decodes genome to architecture
        normal_cell = self._decode_cell(genome[0])
        reduce_cell = self._decode_cell(genome[1])

        return Genotype(
            normal=normal_cell.cell, normal_concat=normal_cell.concat,
            reduce=reduce_cell.cell, reduce_concat=reduce_cell.concat)

    def return_available_ops(self, idx, **kwargs):
        return self.categories[idx]

    def is_valid(self, encode_network: np.ndarray):
        for i in range(0, 32, 4):
            if encode_network[i+1] == encode_network[i+3]:
                return False
        return True

    @staticmethod
    def get_model(genotype):
        return NetworkCIFAR(C=36, num_classes=10, layers=20, auxiliary=True, genotype=genotype)

    # ------------------------ Following functions are specific to DARTS search space ------------------------- #
    @staticmethod
    def _convert_cell(cell_bit_string):
        # convert cell bit-string to genome
        tmp = [cell_bit_string[i:i + 2] for i in range(0, len(cell_bit_string), 2)]
        return [tmp[i:i + 2] for i in range(0, len(tmp), 2)]

    def _convert(self, bit_string):
        # convert network bit-string (norm_cell + reduce_cell) to genome
        norm_gene = self._convert_cell(bit_string[:len(bit_string) // 2])
        reduce_gene = self._convert_cell(bit_string[len(bit_string) // 2:])
        return [norm_gene, reduce_gene]

    def _decode_cell(self, cell_genome):
        cell, cell_concat = [], list(range(2, len(cell_genome) + 2))
        for block in cell_genome:
            for unit in block:
                cell.append((self.allowed_ops[unit[0]], unit[1]))
                # the following lines are for NASNet search space, DARTS simply concat all nodes outputs
                # if unit[1] in cell_concat:
                #     cell_concat.remove(unit[1])
        return Genotype_cell(cell=cell, concat=cell_concat)

if __name__ == '__main__':
    # np.random.seed(0)
    # ss = SS_DARTS()
    # network = ss.sample()
    # genotype = ss.encode(network)
    # print(genotype)
    # print(ss.is_valid(genotype))
    # print(network)
    # genotype = ss.encode(network)
    # print(genotype)
    # print(ss.decode(genotype))
    # print(ss.return_available_ops(0))
    pass