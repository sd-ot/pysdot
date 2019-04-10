from .domain_types import ConvexPolyhedraAssembly
from .radial_funcs import RadialFuncEntropy
from .radial_funcs import RadialFuncUnit
from .PowerDiagram import PowerDiagram
import numpy as np
import importlib


def dist(a, b):
    return np.linalg.norm(a - b, 2)


class OptimalTransport:
    def __init__(self, domain, radial_func=RadialFuncUnit(),
                 obj_max_dw=1e-8, solver="Petsc"):
        self.pd = PowerDiagram(domain, radial_func)
        self.obj_max_dw = obj_max_dw

        self.masses_are_new = True
        self.masses = None

        self.verbosity = 0
        self.max_iter = 1000
        self.delta_w = []
        self.solver = solver

        self.solver_inst = None

    def set_positions(self, new_positions):
        self.pd.set_positions(new_positions)

    def get_positions(self):
        return self.pd.positions

    def set_weights(self, new_weights):
        self.pd.set_weights(new_weights)

    def get_weights(self):
        return self.pd.weights

    def get_masses(self):
        return self.masses

    def set_masses(self, new_masses):
        self.masses_are_new = True
        self.masses = new_masses

    def get_centroids(self):
        return self.pd.centroids()

    def display_vtk(self, filename, points=False, centroids=False):
        self.pd.display_vtk(filename, points, centroids)

    def update_weights(self, ret_if_err=False, relax=1.0):
        if self.masses is None:
            N = self.pd.positions.shape[0]
            self.masses = self.pd.domain.measure() / N * np.ones(N)

        solver = self.get_solver()

        # x = solver.create_vector(size=self.pd.weights.shape[0])

        old_weights = self.pd.weights + 0.0
        for _ in range(self.max_iter):
            # derivatives
            mvs = self.pd.der_integrals_wrt_weights()
            if mvs.error:
                ratio = 0.5
                self.pd.set_weights(
                    (1 - ratio) * old_weights + ratio * self.pd.weights
                )
                if ret_if_err:
                    return True
                print("bim (going back)")
                continue
            old_weights = self.pd.weights

            #
            if self.pd.radial_func.need_rb_corr():
                mvs.m_values[0] *= 2
            mvs.v_values -= self.masses

            A = solver.create_matrix(
                self.pd.weights.shape[0],
                mvs.m_offsets,
                mvs.m_columns,
                mvs.m_values
            )

            b = solver.create_vector(
                mvs.v_values
            )

            x = solver.solve(A, b)

            # update weights
            self.pd.set_weights(self.pd.weights - relax * x)

            nx = np.max(np.abs(x))
            if self.verbosity:
                print("max dw:", nx)

            if nx < self.obj_max_dw:
                break
                
        return False

    def get_solver(self):
        if self.solver_inst is None:
            try:
                mod = importlib.import_module(
                    'pysdot.solvers.{}'.format(self.solver)
                )
            except:
                mod = importlib.import_module(
                    'pysdot.solvers.Scipy'
                )
            self.solver_inst = mod.Solver()
        return self.solver_inst

    def nb_diracs(self):
        return self.pd.positions.shape[0]

    def dim(self):
        return self.pd.positions.shape[1]

    def coalesce_close_diracs(self, min_dist, lst=[]):
        positions = self.get_positions()
        weights = self.get_weights()
        masses = self.masses

        i = 0
        has_change = False
        while i < positions.shape[0]:
            j = i + 1
            while j < positions.shape[0]:
                if dist(positions[i, :], positions[j, :]) < min_dist:
                    print(i, j)
                    masses[i] += masses[j]

                    positions = np.delete(positions, (j), axis=0)
                    weights = np.delete(weights, (j), axis=0)
                    masses = np.delete(masses, (j), axis=0)
                    print("elimination of ", j)
                    has_change = True

                    for k in range(len(lst)):
                        lst[k][i] = 0.5 * (lst[k][i] + lst[k][j])
                        lst[k] = np.delete(lst[k], (j), axis=0)
                else:
                    j += 1
            i += 1

        if has_change:
            self.set_positions(positions)
            self.set_weights(weights)
            self.set_masses(masses)

