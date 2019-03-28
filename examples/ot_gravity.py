from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from pysdot import PowerDiagram

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.linalg import eigvals
import matplotlib.pyplot as plt
import numpy as np
import scipy


def run(n, base_filename, l=0.5):
    # domain
    domain = ConvexPolyhedraAssembly()
    domain.add_box([0, 0], [1, 1])

    # initial positions, weights and masses
    positions = []
    if n == 1:
        radius = 0.2
    else:
        radius = l / (2 * (n - 0))
    for y in np.linspace(radius, l - radius, n):
        for x in np.linspace(radius, l - radius, n):
            nx = x + 0.2 * radius * (np.random.rand() - 0.5)
            ny = y + 0.2 * radius * (np.random.rand() - 0.5)
            positions.append([nx, ny])
    positions = np.array(positions)
    nb_diracs = positions.shape[0]

    # OptimalTransport
    pd = PowerDiagram(domain, RadialFuncInBall())
    pd.set_weights(np.ones(nb_diracs) * radius**2)
    pd.set_positions(positions)

    pd.display_vtk(base_filename + "0.vtk")

    velocity = 0.0 * positions

    for num_iter in range(200):
        print(num_iter)

        mvs = pd.der_centroids_and_integrals_wrt_weight_and_positions()
        m = csr_matrix((mvs.m_values, mvs.m_columns, mvs.m_offsets))
        print( m.todense() )
        print(eigvals(m.todense()))

        # # barycenters at the beginning
        # b_o = pd.get_centroids()

        # # trial for the new barycenters
        # velocity[:, 1] = - 0.05 * radius
        # b_n = b_o + velocity

        break
        # optimisation of positions to go to the target barycenters
        # ropt = scipy.optimize.minimize(
        #     obj,
        #     b_n.flatten(),
        #     (ot, b_n),
        #     tol=1e-4,
        #     method='BFGS',
        #     options={'eps': 1e-4 * radius}
        # )

        # # positions = ropt.x.reshape((-1, 2))
        # ot.set_positions(positions)
        # ot.update_weights()

        # # new barycenters, corrected (minimize have update the weights)
        # b_n = ot.get_centroids()
        # velocity = b_n - b_o
        # print(positions, velocity)

        # # display
        # ot.pd.display_vtk_points(base_filename + "pts_{}.vtk".format(num_iter + 1))
        # ot.display_vtk(base_filename + "{}.vtk".format(num_iter + 1))


run(2, "results/pd_")
