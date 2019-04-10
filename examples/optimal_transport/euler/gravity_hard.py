from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from pysdot import OptimalTransport
import numpy as np


def run(n, base_filename):
    domain = ConvexPolyhedraAssembly()
    domain.add_box([0, 0], [1, 1])
    domain.add_box([0.2, -0.5], [0.8, 0])

    positions = []
    radius = 0.5 / (2 * (n - 1))
    for y in np.linspace(radius, 0.5 + radius, n):
        for x in np.linspace(radius, 0.5 + radius, n):
            positions.append([x, y])
    nb_diracs = len(positions)

    #
    ot = OptimalTransport(domain, RadialFuncInBall())
    ot.set_masses(np.ones(nb_diracs) * 0.8 * 0.5**2 / nb_diracs)
    ot.set_weights(np.ones(nb_diracs) * radius**2)
    ot.set_positions(np.array(positions))
    b_old = ot.pd.centroids()

    ot.adjust_weights()
    ot.display_vtk(base_filename + "0.vtk")

    nb_timesteps = int(20 / radius)
    v = np.zeros((nb_diracs, 2))
    dt = 0.003 * radius
    for i in range(nb_timesteps):
        print(i, "/", nb_timesteps)
        # first trial
        v[:, 1] -= 1

        p_old = ot.get_positions()
        p_tst = p_old + dt * v

        ot.set_positions(p_tst)
        ot.adjust_weights()

        # display
        d = int(n / 5)
        if i % d == 0:
            ot.display_vtk(base_filename + "{:03}.vtk".format(1+int(i / d)))

        # corrections
        b_new = ot.pd.centroids()
        v = (b_new - b_old) / dt
        ot.set_positions(b_new)
        b_old = b_new

run(20, "results/pdhard_")
