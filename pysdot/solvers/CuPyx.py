
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupyx.linalg.sparse.solve import lschol
    
class Solver:
    def __init__(self):
        self.dtype = np.float64

    def create_matrix(self, N, *args):
        (offsets, columns, values) = map(cp.asarray, args)
        return csr_matrix((values, columns, offsets))

    def create_vector(self, values=None, size=0):
        if (values is None):
            return cp.zeros(size, dtype=self.dtype)
        return cp.asarray(values, dtype=self.dtype)

    # solution of Ax = B
    def solve(self, A, b):
        return cp.asnumpy(lschol(A, b))
