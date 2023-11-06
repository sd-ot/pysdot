from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from pysdot import OptimalTransport

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.linalg import eigvals
from scipy.linalg import eig
import matplotlib.pyplot as plt
import numdifftools as nd
import scipy.optimize
import numpy as np
import scipy
import os


def pm( G ):
    print( np.array2string( G.todense(), 5000 ) )


def obj( cx, ot, bh, dt ):
    op = ot.get_positions() + 0.0
    pc = cx.reshape( ( -1, 2 ) )
    ot.set_positions( pc )
    ot.adjust_weights()

    bm = np.array( bh[ -2 ].flat )
    b0 = np.array( bh[ -1 ].flat )
    bt = 2 * b0 - bm

    bc = np.array( ot.get_centroids().flat )
    dlt = bc - bt

    ot.set_positions( op )
    return 0.5 * np.sum( dlt ** 2 )


def fit_positions( ot, bh, dt ):
    for num_iter in range( 1000 ):
        X = np.array( ot.get_positions().flat )
        fun = lambda cx: obj( cx, ot, bh, dt )
        g = nd.Gradient( fun, 1e-6 )
        h = nd.Hessian( fun, 1e-6 )
        M = np.array( h( X ) )
        V = np.array( g( X ) )

        d = np.linalg.solve( M, V )
        norm = np.linalg.norm( d, ord=np.inf )
        if norm > 1e-3:
            d *= 1e-3 / norm
        print( "  sub_iter:", num_iter, "norm:", norm )

        ot.set_positions( ot.get_positions() - d.reshape( ( -1, 2 ) ) )
        ot.adjust_weights()

        # error
        bm = np.array( bh[ -2 ].flat )
        b0 = np.array( bh[ -1 ].flat )
        bt = 2 * b0 - bm

        bc = np.array( ot.get_centroids().flat )
        dlt = bc - bt

        ot.set_positions( op )
        return 0.5 * np.sum( dlt ** 2 )


def run( n, base_filename, l=0.5 ):
    # domain
    domain = ConvexPolyhedraAssembly()
    domain.add_box( [ 0, 0 ], [ 1, 1 ] )

    # initial positions, weights and masses
    positions = []
    mass = l**2 / n**2
    radius = l / ( 2 * ( n - 1 ) )
    for y in np.linspace( radius, l - radius, n ):
        for x in np.linspace( 0.5 - l / 2 + radius, 0.5 + l / 2 - radius, n ):
            nx = x + 0.0 * radius * ( np.random.rand() - 0.5 )
            ny = y + 0.0 * radius * ( np.random.rand() - 0.5 )
            positions.append([nx, ny])
    positions = np.array(positions)
    nb_diracs = positions.shape[ 0 ]

    # OptimalTransport
    ot = OptimalTransport( domain, RadialFuncInBall() )
    ot.set_weights( np.ones( nb_diracs ) * radius**2 )
    ot.set_masses( np.ones( nb_diracs ) * mass )
    ot.set_positions( positions )
    ot.max_iter = 500
    ot.adjust_weights()

    ot.display_vtk( base_filename + "0.vtk", points=True, centroids=True )

    # history of centroids
    ce = ot.get_centroids()
    ce[ :, 1 ] += radius / 10
    bh = [ce]

    dt = 1.0
    for num_iter in range( 50 ):
        print( "num_iter", num_iter )

        bh.append( ot.get_centroids() )
        fit_positions( ot, bh, dt )

        # display
        n1 = int( num_iter / 1 ) + 1
        ot.display_vtk( base_filename + "{}.vtk".format( n1 ), points=True, centroids=True )


os.system( "rm results/hd_*" )
run( 5, "results/hd_" )
