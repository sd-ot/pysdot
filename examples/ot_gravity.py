from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.examples import FluidSystem
import matplotlib.pyplot as plt
import numpy as np
import os


def run(n, base_filename, l=0.5):
    os.system("rm {}*.vtk".format(base_filename))

    # initial conditions
    domain = ConvexPolyhedraAssembly()
    domain.add_box([0, 0], [1, 1])

    positions = []
    velocities = []
    radius = 0.5 * l / n
    masses = np.ones(n**2) * l**2 / n**2
    for y in np.linspace(radius, l - radius, n):
        for x in np.linspace(0.5 - l / 2 + radius, 0.5 + l / 2 - radius, n):
            nx = x + 0 * radius * (np.random.rand() - 0.5)
            ny = y + 0 * radius * (np.random.rand() - 0.5)
            velocities.append([0, -radius/5])
            positions.append([nx, ny])


    #   
    fs = FluidSystem(domain, positions, velocities, masses, base_filename)
    fs.display()

    # for num_iter in range(10):
    #     print(num_iter)

    #     s.get_new_positions_for_centroids()
    #     s.get_new_weights_for_areas()
    #     s.display()

#
run(10, "results/pd_")
