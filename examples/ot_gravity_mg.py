from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from pysdot import OptimalTransport

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.linalg import eigvals
from scipy.linalg import eig
import matplotlib.pyplot as plt
import scipy.optimize
import numpy as np
import scipy
import os

def run( n, base_filename, l=0.5 ):
    # domain
    domain = ConvexPolyhedraAssembly()
    domain.add_box( [ 0, 0 ], [ 1, 1 ] )

    # initial positions, weights and masses
    positions = []
    radius = l / ( 2 * ( n - 1 ) )
    mass = l**2 / n**2
    for y in np.linspace( radius, l - radius, n ):
        for x in np.linspace( 0.5 - l / 2 + radius, 0.5 + l / 2 - radius, n ):
            nx = x + 0.0 * radius * ( np.random.rand() - 0.5 )
            ny = y + 0.0 * radius * ( np.random.rand() - 0.5 )
            positions.append( [ nx, ny ] )
    positions = np.array( positions )
    nb_diracs = positions.shape[ 0 ]
    dim = positions.shape[ 1 ]

    # OptimalTransport
    ot = OptimalTransport( domain, RadialFuncInBall() )
    ot.set_weights( np.ones( nb_diracs ) * radius**2 )
    ot.set_masses( np.ones( nb_diracs ) * mass )
    ot.set_positions( positions )
    ot.max_iter = 100

    ot.update_weights()
    ot.display_vtk( base_filename + "0.vtk", points=True, centroids=True )

    # gravity
    G = np.zeros( ( nb_diracs, dim ) )
    G[ :, 1 ] = -9.81

    # 
    eps = 0.5
    dt = radius * 0.1
    V = np.zeros( ( nb_diracs, dim ) )
    M = np.stack( [ ot.get_masses() for d in range( dim ) ] ).transpose()
    for num_iter in range( 500 ):
        print( "num_iter:", num_iter, "dt:", dt )
        C = ot.get_centroids()
        X = ot.get_positions()

        A = G + ( C - ot.get_positions() ) / ( M * eps**2 )

        while True:
            dV = dt * A
            dX = dt * ( V + dV )
            if np.max( np.linalg.norm( dX, axis=1, ord=2 ) ) < 0.2 * radius:
                dt *= 1.05
                V += dV
                X += dX
                break
            dt *= 0.5

        ot.set_positions( X )
        ot.update_weights()

        # display
        n1 = int( num_iter / 1 ) + 1
        ot.display_vtk( base_filename + "{}.vtk".format( n1 ), points=True, centroids=True )


os.system( "rm results/mg_*" )
run( 10, "results/mg_" )
