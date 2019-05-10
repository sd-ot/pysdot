from .domain_types import ConvexPolyhedraAssembly
from .radial_funcs import RadialFuncEntropy
from .radial_funcs import RadialFuncInBall
from .radial_funcs import RadialFuncUnit
from .PowerDiagram import PowerDiagram
import numpy as np
import importlib


def dist(a, b):
    return np.linalg.norm(a - b, 2)


class OptimalTransport:
    def __init__(self, positions=None, weights=None, domain=None, radial_func=RadialFuncUnit(),
                 obj_max_dw=1e-8, linear_solver="Petsc"):
        self.pd = PowerDiagram(positions, weights, domain, radial_func)
        self.obj_max_dw = obj_max_dw

        self.masses = None

        self.linear_solver = linear_solver
        self.verbosity = 0
        self.max_iter = 1000
        self.delta_w = []

        self._linear_solver_inst = None
        self._masses_are_new = True

    def get_positions(self):
        return self.pd.positions

    def set_positions(self, new_positions):
        self.pd.set_positions(new_positions)

    def get_masses(self):
        return self.masses

    def set_masses(self, new_masses):
        self._masses_are_new = True
        self.masses = new_masses

    def get_domain(self):
        return self.pd.get_domain()

    def set_domain(self, new_domain):
        self.pd.set_domain(new_domain)

    def get_weights(self):
        return self.pd.weights

    def set_weights(self, new_weights):
        self.pd.set_weights(new_weights)

    def adjust_weights(self, initial_weights=None, ret_if_err=False, relax=1.0):
        if not ( initial_weights is None ):
            self.set_weights( initial_weights )
            
        if self.pd.domain is None:
            domain = ConvexPolyhedraAssembly()
            domain.add_box([0, 0], [1, 1])
            self.pd.set_domain( domain )

        if self.masses is None:
            if isinstance(self.pd.radial_func, RadialFuncUnit):
                N = self.pd.positions.shape[0]
                self.masses = np.ones(N) * self.pd.domain.measure() / N
            elif isinstance(self.pd.radial_func, RadialFuncInBall):
                self.masses = np.ones(N) * 1e-6
            else:
                TODO

        if self.pd.weights is None:
            self.pd.weights = np.sqrt( self.masses )

        linear_solver = self._get_linear_solver()
        old_weights = self.pd.weights + 0.0
        for _ in range(self.max_iter):
            # derivatives
            mvs = self.pd.der_integrals_wrt_weights(stop_if_void=True)
            if mvs.error:
                ratio = 0.5
                self.pd.set_weights(
                    (1 - ratio) * old_weights + ratio * self.pd.weights
                )
                if ret_if_err:
                    return True
                if self.verbosity:
                    print("bim (going back)")
                continue
            old_weights = self.pd.weights

            #
            if self.pd.radial_func.need_rb_corr():
                mvs.m_values[0] *= 2
            mvs.v_values -= self.masses

            A = linear_solver.create_matrix(
                self.pd.weights.shape[0],
                mvs.m_offsets,
                mvs.m_columns,
                mvs.m_values
            )

            b = linear_solver.create_vector(
                mvs.v_values
            )

            x = linear_solver.solve(A, b)

            # update weights
            self.pd.set_weights(self.pd.get_weights() - relax * x)

            nx = np.max(np.abs(x))
            if self.verbosity:
                print("max dw:", nx)

            if nx < self.obj_max_dw:
                break
                
        return False

    def get_centroids(self):
        return self.pd.centroids()

    def display_vtk(self, filename, points=False, centroids=False):
        self.pd.display_vtk(filename, points, centroids)

    def display_asy(self, filename, preamble="", closing="", output_format="pdf", linewidth=0.02, dotwidth=0.0, values=np.array([]), colormap="inferno", avoid_bounds=False, min_rf=1, max_rf=0):
        self.pd.display_asy(filename, preamble, closing, output_format, linewidth, dotwidth, values, colormap, avoid_bounds, min_rf, max_rf)

    def nb_diracs(self):
        return self.pd.positions.shape[0]

    def dim(self):
        return self.pd.positions.shape[1]

    def _get_linear_solver(self):
        if self._linear_solver_inst is None:
            try:
                mod = importlib.import_module(
                    'pysdot.solvers.{}'.format(self.linear_solver)
                )
            except:
                mod = importlib.import_module(
                    'pysdot.solvers.Scipy'
                )
            self._linear_solver_inst = mod.Solver()
        return self._linear_solver_inst
