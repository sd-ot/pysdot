from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from pysdot import OptimalTransport
import numpy as np

# constants
for n in [60]:
    directory = "results/"  # .format(n)

    #  domain.display_boundaries_vtk( directory + "/bounds.vtk" )
    domain = ConvexPolyhedraAssembly()
    domain.add_box([0, 0], [1, 1])

    radius = 0.5 / n
    positions = []
    for y in np.arange(radius, 0.5, 2*radius):
        for x in np.arange(radius, 0.5, 2*radius):
            positions.append([x, y])
    nb_diracs = len(positions)

    #
    ot = OptimalTransport(domain, RadialFuncInBall())
    ot.set_masses(np.ones(nb_diracs) * np.pi * radius**2)
    ot.set_weights(np.ones(nb_diracs) * radius**2)
    ot.set_positions(np.array(positions))
    b_old = ot.pd.centroids()

    nb_timesteps = int(20 / radius)
    v = np.zeros((nb_diracs, 2))
    dt = 0.003 * radius
    for i in range(nb_timesteps):
        # first trial
        v[:, 1] -= 1

        p_old = ot.get_positions()
        p_tst = p_old + dt * v

        ot.set_positions(p_tst)
        ot.update_weights()

        # display
        d = int(n / 5)
        if i % d == 0:
            ot.pd.display_vtk(directory + "/pd_{:03}.vtk".format(int(i / d)))

        # corrections
        b_new = ot.pd.centroids()
        v = (b_new - b_old) / dt
        ot.set_positions(b_new)
        b_old = b_new
