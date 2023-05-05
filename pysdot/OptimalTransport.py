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
    def __init__(self, positions=None, weights=None, domain=None, masses=None, radial_func=RadialFuncUnit(),
                 obj_max_dw=1e-8, obj_max_dm=0, linear_solver="Petsc", verbosity=0):
        """
           stopping criterion = first obj_max_xy that is != 0
             * obj_max_dw => delta weights between two iterations
             * obj_max_dm => 
        """

        self.pd = PowerDiagram(positions, weights, domain, radial_func)
        self.obj_max_dw = obj_max_dw
        self.obj_max_dm = obj_max_dm
        self.masses = masses
        
        self.linear_solver = linear_solver
        self.verbosity = verbosity
        self.max_iter = 1000
        self.delta_m = []
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
        assert( self.obj_max_dw or self.obj_max_dm )

        if not ( initial_weights is None ):
            self.set_weights( initial_weights )
            
        if self.pd.domain is None:
            domain = ConvexPolyhedraAssembly()
            if self.pd.positions.shape[ 1 ] == 2:
                domain.add_box([0, 0], [1, 1])
            else:
                domain.add_box([0, 0, 0], [1, 1, 1])
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
                if (self.verbosity > 1):
                    print("bim (going back)")
                continue
            old_weights = self.pd.weights

            #
            if self.pd.radial_func.need_rb_corr():
                mvs.m_values[0] *= 2
            mvs.v_values -= self.masses

            # "dm" stopping criterion
            nm = np.max(np.abs(mvs.v_values))
            self.delta_m.append(nm)
            if self.obj_max_dm:
                if self.verbosity > 1:
                    print("max dm:", nm)
                if nm < self.obj_max_dm:
                    break

            # linear system
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
            loc_relax = relax
            cpt_loc = 0
            while True:
                W = self.pd.get_weights() - loc_relax * x
                if self.pd.radial_func.ball_cut() == False or np.all( W >= 0 ): # HUM
                    self.pd.set_weights( W )
                    break
                if self.verbosity > 1:
                    print("negative weight, loc_relax=", loc_relax)
                loc_relax *= 0.75

                cpt_loc += 1
                if cpt_loc == 50:
                    print( "impossible to get positive weights" )
                    return True

            # "dw" stopping criterion
            nw = np.max(np.abs(x))
            self.delta_w.append(nw)
            if self.obj_max_dw:
                if self.verbosity > 1:
                    print("max dw:", nw)
                if nw < self.obj_max_dw:
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

    def set_stopping_criterion(self, value, type="max delta masses"):
        """
            Possible values for type
            * "max delta masses" => max(abs(actual masses - target ones))
            * "max delta weights" => max(abs(weights - weights last iteration))
        """

        self.obj_max_dw = 0
        self.obj_max_dm = 0

        if type == "max delta masses":
            self.obj_max_dm = value
            return

        if type == "max delta weights":
            self.obj_max_dw = value
            return

        raise "'{}' is not a known stopping criterion type".format( type )

    def _get_linear_solver(self):
        if (self._linear_solver_inst is not None):
            return self._linear_solver_inst
        
        sparse_linear_solvers = ('CuPyx', 'Petsc', 'Scipy')
        msg = 'Available solvers are: {}.'.format(', '.join(sparse_linear_solvers))
        assert self.linear_solver in sparse_linear_solvers, msg

        for solver in (self.linear_solver,)+sparse_linear_solvers:
            try:
                module = importlib.import_module('pysdot.solvers.{}'.format(solver))
                break
            except ImportError:
                continue
        else:
            msg='Could not import any of the solver modules.'
            raise ImportError(msg)

        if (self.verbosity > 0):
            print('Sucessfully imported sparse linear solver {}.'.format(solver))

        self._linear_solver_inst = module.Solver()
        return self._linear_solver_inst

