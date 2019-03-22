from petsc4py import PETSc


#
class Solver:
    def __init__(self):
        pass

    def create_matrix(self, N, offsets, columns, values):
        A = PETSc.Mat().createAIJ([N, N], csr=(
            offsets.astype(PETSc.IntType),
            columns.astype(PETSc.IntType),
            values
        ))
        A.assemblyBegin()
        A.assemblyEnd()

        return A

    def create_vector(self, values=None, size=0):
        if values is None:
            return PETSc.Vec().createSeq(size)
        return PETSc.Vec().createWithArray(values)

    # solution of Ax = B
    def solve(self, A, b):
        size = A.getSizes()[0]
        x = PETSc.Vec().createSeq(size)

        # Initialize ksp solver.
        ksp = PETSc.KSP().create()
        ksp.setType('cg')
        ksp.getPC().setType('gamg')

        ksp.setOperators(A)
        ksp.setFromOptions()

        # Solve
        ksp.solve(b, x)

        return x
        