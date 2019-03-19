from .domain_types import ConvexPolyhedraAssembly
from .radial_funcs import RadialFuncEntropy
from .radial_funcs import RadialFuncUnit
from .PowerDiagram import PowerDiagram

from petsc4py import PETSc
import numpy as np


class OptimalTransport:
    def __init__(self, domain, radial_func=RadialFuncUnit(), obj_max_dw=1e-7):
        self.pd = PowerDiagram(domain, radial_func)
        self.obj_max_dw = obj_max_dw

        self.masses_are_new = True
        self.masses = None

        self.max_iter = 10
        self.delta_w = []

    def set_positions(self, new_positions):
        self.pd.set_positions(new_positions)

    def get_positions(self):
        return self.pd.positions

    def set_weights(self, new_weights):
        self.pd.set_weights(new_weights)

    def get_weights(self):
        return self.pd.weights

    def set_masses(self, new_masses):
        self.masses_are_new = True
        self.masses = new_masses

    def update_weights(self):
        if self.masses is None:
            N = self.pd.positions.shape[0]
            self.masses = self.pd.domain.measure() / N * np.ones(N)

        x = PETSc.Vec().createSeq(self.pd.weights.shape[0])

        old_weights = self.pd.weights + 0.0
        for _ in range(self.max_iter):
            # derivatives
            mvs = self.pd.der_integrals_wrt_weights()
            if mvs.error:
                ratio = 0.5
                self.pd.set_weights(
                    (1 - ratio) * old_weights + ratio * self.pd.weights
                )
                print("bim (going back)")
                continue
            old_weights = self.pd.weights

            #
            if self.pd.radial_func.need_rb_corr():
                mvs.m_values[0] *= 2
            mvs.v_values -= self.masses

            A = PETSc.Mat().createAIJ(
                [self.pd.weights.shape[0], self.pd.weights.shape[0]], 
                csr=(
                    mvs.m_offsets.astype(PETSc.IntType), 
                    mvs.m_columns.astype(PETSc.IntType),
                    mvs.m_values
                )
            )
            b = PETSc.Vec().createWithArray(mvs.v_values)
            A.assemblyBegin()  # Make matrices useable.
            A.assemblyEnd()

            # Initialize ksp solver.
            ksp = PETSc.KSP().create()
            ksp.setType('cg')
            ksp.getPC().setType('gamg')

            ksp.setOperators(A)
            ksp.setFromOptions()

            # Solve
            ksp.solve(b, x)

            # update weights
            self.pd.set_weights(self.pd.weights - x)

            nx = np.max(np.abs(x))
            print("max dw:", nx)

            if nx < self.obj_max_dw:
                break
