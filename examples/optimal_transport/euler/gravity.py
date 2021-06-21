from pysdot.domain_types import ConvexPolyhedraAssembly
import matplotlib.pyplot as plt
import FluidSystem
import numpy as np
import os

import matplotlib.pyplot as plt

def run(n, base_filename, l=0.5):
    os.system("rm {}*.vtk".format(base_filename))

    # initial conditions
    domain = ConvexPolyhedraAssembly()
    domain.add_box( [ 0, 0 ], [ 1, 1 ] )

    positions = []
    velocities = []
    radius = 0.5 * l / n
    for y in np.linspace( radius, l - radius, n ):
        for x in np.linspace( 0.5 - l / 2 + radius, 0.5 + l / 2 - radius, n ):
            nx = x + 0 * radius * ( np.random.rand() - 0.5 )
            ny = y + 0 * radius + 0 * radius * ( np.random.rand() - 0.5 )
            if ( nx - 0.5 )**2 + ( ny - 0.5 * l )**2 < ( 0.5 * l )**2:
                velocities.append( [ 0, -radius/5 ] )
                positions.append( [ nx, ny ] )
    masses = np.ones( len( positions ) ) * l**2 / n**2

    # simulation
    fs = FluidSystem.FluidSystem( domain, positions, velocities, masses, base_filename )
    fs.coeff_centroid_force = 1e-5
    fs.display()

    for num_iter in range( 500 ):
        print( "num_iter:", num_iter, "time:", fs.time )
        fs.make_step()
        fs.display()

#
run(20, "results/pd_")
