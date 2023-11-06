from .domain_types import ConvexPolyhedraAssembly
from .OptimalTransport import OptimalTransport
from .PowerDiagram import PowerDiagram
import numpy as np
import importlib

class TransportMap:
    def __init__(self, positions, masses=None, initial_weights=None, domain=None) -> None:
        if domain is None:
            domain = ConvexPolyhedraAssembly()
            if positions.shape[ 1 ] == 2:
                domain.add_box([0, 0], [1, 1])
            elif positions.shape[ 1 ] == 3:
                domain.add_box([0, 0, 0], [1, 1, 1])
            else:
                raise "TODO"
            
        if masses is None:
            N = positions.shape[ 0 ]
            masses = np.ones(N) * domain.measure() / N

        self.pd = PowerDiagram(positions=positions, weights=initial_weights, domain=domain)
        self.masses = masses

        self.obj_max_dw = 1e-6
        self.max_iter = 100
        self.verbosity = 2

        self._linear_solver_inst = None
        self.linear_solver = "Petsc"

    def optimize_weights(self) -> None:
        linear_solver = self._get_linear_solver()
        old_weights = np.copy(self.pd.weights)
        relax = 1
        for _ in range(self.max_iter):
            # derivatives
            mvs = self.pd.der_integrals_wrt_weights(stop_if_void=True)
            if mvs.error:
                ratio = 0.5
                self.pd.set_weights((1 - ratio) * old_weights + ratio * self.pd.weights)
                if self.verbosity > 1:
                    print("bim (going back)")
                continue
            old_weights = self.pd.weights

            #
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

            # update weights. Attention aux poids négatifs
            self.pd.set_weights(self.pd.get_weights() - relax * x)

            nx = np.max(np.abs(x))
            if self.verbosity > 1:
                print("max dw:", nx)
            if nx < self.obj_max_dw:
                break
                
        return False
    

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
            msg='Could not import any of the solver modules. `pip install scipy` will install the most simple one. Otherwise you can install `cupy` or `petsc4py` for more advanced solutions'
            raise ImportError(msg)

        if self.verbosity > 0:
            print('Successfully imported sparse linear solver {}.'.format(solver))

        self._linear_solver_inst = module.Solver()
        return self._linear_solver_inst

    def show(self, arrows=False, arrow_width=0.05, line_width_cells=10, line_width_arrows=10):
        import pyvista as pv
        from pyvista import themes
        pv.global_theme.anti_aliasing = 'ssaa'
        # pv.set_plot_theme(themes.ParaViewTheme())
        # pv.global_theme.depth_peeling.enabled = True

        cells, celltypes, points = self.pd.vtk_mesh_data(0.01)
        m = pv.UnstructuredGrid(cells, celltypes, points) # .shrink(0.999)

        p = pv.Plotter()
        p.add_mesh(m, show_edges=True, line_width=line_width_cells, style="wireframe")

        # 2D case ?
        if self.pd.positions.shape[1] == 2:
            mi = np.min( self.pd.positions, axis = 0 )
            ma = np.max( self.pd.positions, axis = 0 )
            me = ( mi + ma ) / 2
            di = ma - mi

            p.set_focus([me[0], me[1], 0])
            p.set_position([me[0], me[1], 3 * np.max(di)])
            p.set_viewup([0, 1, 0])

        # arrows
        if arrows:
            positions = self.pd.get_positions()
            centroids = self.pd.centroids()
            N = positions.shape[0]
            m = arrow_width * np.mean(np.linalg.norm(centroids - positions,axis=1))
            for n in range(N):
                pos = np.copy(positions[n, :])
                cen = np.copy(centroids[n, :])
                pos.resize(3)
                cen.resize(3)

                dy = m * ( cen - pos ) / np.linalg.norm( cen - pos )
                dx = np.array([ -dy[ 1 ], dy[ 0 ], 0 ])

                # pts = [ pos, cen - dy, cen - dy + 0.5 * dx, cen, cen - dy - 0.5 * dx, cen - dy ]
                pts = [ 
                    pos + 0.2 * dy,
                    pos - 0.2 * dx,
                    pos - 0.2 * dy,
                    pos + 0.2 * dx,
                    pos + 0.2 * dy, 
                    cen - dy,
                    cen - dy + 0.5 * dx, 
                    cen, 
                    cen - dy - 0.5 * dx, 
                    cen - dy
                ]

                p.add_mesh(pv.MultipleLines(pts), color="red", line_width=line_width_arrows)

        p.set_background("white")
        p.show()

    def write_vtk(self, filename, points=False, centroids=False):
        self.pd.display_vtk(filename, points, centroids)


def find_transport_map(positions, masses=None, initial_weights=None, domain=None):
    res = TransportMap(positions, masses, initial_weights, domain)
    res.optimize_weights()
    return res
