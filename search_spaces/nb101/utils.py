import hashlib
import copy
import numpy as np

def hash_module(matrix, labeling):
    """Computes a graph-invariance MD5 hash of the matrix and label pair.

    Args:
      matrix: np.ndarray square upper-triangular adjacency matrix.
      labeling: list of int labels of length equal to both dimensions of
        matrix.

    Returns:
      MD5 hash of the matrix and labeling.
    """
    vertices = np.shape(matrix)[0]
    in_edges = np.sum(matrix, axis=0).tolist()
    out_edges = np.sum(matrix, axis=1).tolist()

    assert len(in_edges) == len(out_edges) == len(labeling)
    hashes = list(zip(out_edges, in_edges, labeling))
    hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
    # Computing this up to the diameter is probably sufficient but since the
    # operation is fast, it is okay to repeat more times.
    for _ in range(vertices):
        new_hashes = []
        for v in range(vertices):
            in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
            out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
            new_hashes.append(hashlib.md5(
                (''.join(sorted(in_neighbors)) + '|' +
                 ''.join(sorted(out_neighbors)) + '|' +
                 hashes[v]).encode('utf-8')).hexdigest())
        hashes = new_hashes
    fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()
    return fingerprint

class ModelSpec(object):
    """Model specification given adjacency matrix and labeling."""

    def __init__(self, matrix, ops, data_format='channels_last'):
        """Initialize the module spec.

        Args:
          matrix: ndarray or nested list with shape [V, V] for the adjacency matrix.
          ops: V-length list of labels for the base ops used. The first and last
            elements are ignored because they are the input and output vertices
            which have no operations. The elements are retained to keep consistent
            indexing.
          data_format: channels_last or channels_first.

        Raises:
          ValueError: invalid matrix or ops
        """
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        shape = np.shape(matrix)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('matrix must be square')
        if shape[0] != len(ops):
            raise ValueError('length of ops must match matrix dimensions')
        if not is_upper_triangular(matrix):
            raise ValueError('matrix must be upper triangular')

        # Both the original and pruned matrices are deep copies of the matrix and
        # ops so any changes to those after initialization are not recognized by the
        # spec.
        self.original_matrix = copy.deepcopy(matrix)
        self.original_ops = copy.deepcopy(ops)

        self.matrix = copy.deepcopy(matrix)
        self.ops = copy.deepcopy(ops)
        self.valid_spec = True
        self._prune()

        self.data_format = data_format

    def _prune(self):
        """Prune the extraneous parts of the graph.

        General procedure:
          1) Remove parts of graph not connected to input.
          2) Remove parts of graph not connected to output.
          3) Reorder the vertices so that they are consecutive after steps 1 and 2.

        These 3 steps can be combined by deleting the rows and columns of the
        vertices that are not reachable from both the input and output (in reverse).
        """
        num_vertices = np.shape(self.original_matrix)[0]

        # DFS forward from input
        visited_from_input = set([0])
        frontier = [0]
        while frontier:
            top = frontier.pop()
            for v in range(top + 1, num_vertices):
                if self.original_matrix[top, v] and v not in visited_from_input:
                    visited_from_input.add(v)
                    frontier.append(v)

        # DFS backward from output
        visited_from_output = set([num_vertices - 1])
        frontier = [num_vertices - 1]
        while frontier:
            top = frontier.pop()
            for v in range(0, top):
                if self.original_matrix[v, top] and v not in visited_from_output:
                    visited_from_output.add(v)
                    frontier.append(v)

        # Any vertex that isn't connected to both input and output is extraneous to
        # the computation graph.
        extraneous = set(range(num_vertices)).difference(
            visited_from_input.intersection(visited_from_output))

        # If the non-extraneous graph is less than 2 vertices, the input is not
        # connected to the output and the spec is invalid.
        if len(extraneous) > num_vertices - 2:
            self.matrix = None
            self.ops = None
            self.valid_spec = False
            return

        self.matrix = np.delete(self.matrix, list(extraneous), axis=0)
        self.matrix = np.delete(self.matrix, list(extraneous), axis=1)
        for index in sorted(extraneous, reverse=True):
            del self.ops[index]

    def hash_spec(self, canonical_ops):
        """Computes the isomorphism-invariant graph hash of this spec.

        Args:
          canonical_ops: list of operations in the canonical ordering which they
            were assigned (i.e. the order provided in the config['available_ops']).

        Returns:
          MD5 hash of this spec which can be used to query the dataset.
        """
        # Invert the operations back to integer label indices used in graph gen.
        labeling = [-1] + [canonical_ops.index(op) for op in self.ops[1:-1]] + [-2]
        return hash_module(self.matrix, labeling)


def is_upper_triangular(matrix):
    """True if matrix is 0 on diagonal and below."""
    for src in range(np.shape(matrix)[0]):
        for dst in range(0, src + 1):
            if matrix[src, dst] != 0:
                return False
    return True

class OutOfDomainError(Exception):
    """Indicates that the requested graph is outside of the search domain."""

ModelSpec_ = ModelSpec

def check_spec(model_spec):
    """Checks that the model spec is within the dataset."""
    if not model_spec.valid_spec:
        raise OutOfDomainError('invalid spec, provided graph is disconnected.')

    num_vertices = len(model_spec.ops)
    num_edges = np.sum(model_spec.matrix)

    if num_vertices > 7:
        raise OutOfDomainError('too many vertices')

    if num_edges > 9:
        raise OutOfDomainError('too many edges')

    if model_spec.ops[0] != 'input':
        raise OutOfDomainError('first operation should be \'input\'')

    if model_spec.ops[-1] != 'output':
        raise OutOfDomainError('last operation should be \'output\'')

    for op in model_spec.ops[1:-1]:
        if op not in ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']:
            raise OutOfDomainError('unsupported op')