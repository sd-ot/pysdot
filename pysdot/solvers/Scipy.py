from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import numpy as np
    

#
class Solver:
    def __init__(self):
        pass

    def create_matrix(self, N, offsets, columns, values):
        return csr_matrix((values, columns, offsets))

    def create_vector(self, values=None, size=0):
        if values is None:
            return np.zeros(size)
        return np.array(values)

    # solution of Ax = B
    def solve(self, A, b):
        return spsolve(A, b)
