from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from pysdot import OptimalTransport
import matplotlib.pyplot as plt
import scipy.optimize
import numpy as np


# objective function
def obj(x, ot, new_barycenters):
    # ot.set_weights(ot.get_weights()*0 + 1e-3)
    pos = x.reshape((-1, 2))
    ot.set_positions(pos)
    ot.update_weights()

    prp = ot.get_centroids()
    dlt = new_barycenters - prp
    dlp = pos - new_barycenters
    return np.sum(dlt**2) \
        + 1e-2 * np.sum(dlp**2)


def run(n, base_filename, l=0.5):
    # domain
    domain = ConvexPolyhedraAssembly()
    domain.add_box([0, 0], [1, 1])

    # initial positions, weights and masses
    positions = []
    if n == 1:
        radius = 0.2
    else:
        radius = l / (2 * (n - 1))
    for y in np.linspace(radius, l - radius, n):
        for x in np.linspace(radius, l - radius, n):
            nx = x + 0.2 * radius * (np.random.rand() - 0.5)
            ny = y + 0.2 * radius * (np.random.rand() - 0.5)
            positions.append([nx, ny])
    positions = np.array(positions)
    nb_diracs = positions.shape[0]

    # OptimalTransport
    ot = OptimalTransport(domain, RadialFuncInBall())
    ot.set_weights(np.ones(nb_diracs) * radius**2)
    ot.set_masses(np.ones(nb_diracs) * l**2 / nb_diracs)
    ot.set_positions(positions)
    ot.update_weights()

    ot.display_vtk(base_filename + "0.vtk")

    ot.set_positions(ot.get_centroids())

    velocity = 0.0 * positions

    for num_iter in range(200):
        print(num_iter)

        # barycenters at the beginning
        ot.update_weights()
        b_o = ot.get_centroids()

        # trial for the new barycenters
        velocity[:, 1] = - 0.05 * radius
        b_n = b_o + velocity

        # optimisation of positions to go to the target barycenters
        ropt = scipy.optimize.minimize(
            obj,
            b_n.flatten(),
            (ot, b_n),
            tol=1e-4,
            method='BFGS',
            options={'eps': 1e-4 * radius}
        )

        positions = ropt.x.reshape((-1, 2))
        ot.set_positions(positions)
        ot.update_weights()

        # new barycenters, corrected (minimize have update the weights)
        b_n = ot.get_centroids()
        velocity = b_n - b_o
        print(positions, velocity)

        # display
        ot.pd.display_vtk_points(base_filename + "pts_{}.vtk".format(num_iter + 1))
        ot.display_vtk(base_filename + "{}.vtk".format(num_iter + 1))


run(5, "results/pd_")
