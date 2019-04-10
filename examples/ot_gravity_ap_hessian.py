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
    ot.update_weights()

    bm = np.array( bh[ -2 ].flat )
    b0 = np.array( bh[ -1 ].flat )
    bc = np.array( ot.get_centroids().flat )
    bt = 2 * b0 - bm

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
        if norm > 1e-4:
            d *= 1e-4 / norm
        print( "  num_iter:", num_iter, "norm:", norm )

        ot.set_positions( ot.get_positions() - d.reshape( ( -1, 2 ) ) )
        ot.update_weights()


    # # get G
    # mvs = ot.pd.der_centroids_and_integrals_wrt_weight_and_positions()
    # print( mvs.error )
    # m = csr_matrix((mvs.m_values, mvs.m_columns, mvs.m_offsets))

    # rd = np.arange( dim * nb_diracs, dtype=np.int )
    # b0 = ( dim + 1 ) * np.floor_divide( rd, dim )
    # l0 = b0 + rd % dim 
    # l1 = ( dim + 1 ) * np.arange( nb_diracs, dtype=np.int) + dim
    
    # C = m[l0, :][:, l0]
    # D = m[l0, :][:, l1]
    # F = m[l1, :][:, l1]
    # E = m[l1, :][:, l0]

    # G = C - D * spsolve( F.tocsc(), E.tocsc() )
    # print( eig( ( G.transpose() * G ).todense() ) )
    # pm( G.transpose() * G )

    # for d in range( dim ):
    #     V = np.zeros( dim * nb_diracs )
    #     V[ dim * 4 + d ] = 1
    #     print( np.array2string( G * V, 5000 ) )

    # # get G
    # G = np.zeros( ( nb_diracs * dim, nb_diracs * dim ) )
    # ref_positions = ot.get_positions() + 0.0
    # ref_centroids = ot.get_centroids() + 0.0
    # eps = 1e-6
    # for i in range( nb_diracs * dim ):
    #     positions = ref_positions + 0.0
    #     positions[ int( i / dim ), i % dim ] += eps
    #     ot.set_positions( positions )
    #     ot.update_weights()

    #     G[ i, : ] = ( ( ot.get_centroids() - ref_centroids ) / eps ).flat
    # ot.set_positions( ref_positions )
    # ot.update_weights()

    # # centroids
    # # pc = cx.reshape( ( -1, 2 ) )
    # # ot.set_positions( pc )

    # bm = np.array( bh[ -2 ].flat )
    # b0 = np.array( bh[ -1 ].flat )
    # bc = np.array( ot.get_centroids().flat )
    # bt = 2 * b0 - bm



    # ropt = scipy.optimize.minimize(
    #     obj,
    #     ot.get_positions().flatten(),
    #     ( ot, bh, dt ),
    #     tol=1e-6,
    #     method='BFGS',
    #     # jac=jac
    # )
    # print(ropt.njev)

    # positions = ropt.x.reshape((-1, 2))
    # ot.set_positions(positions)
    # ot.update_weights()


def run( n, base_filename, l=0.5 ):
    # domain
    domain = ConvexPolyhedraAssembly()
    domain.add_box( [ 0, 0 ], [ 1, 1 ] )

    # initial positions, weights and masses
    positions = []
    if n == 1:
        radius = 0.3
        mass = 3.14159 * radius**2
        positions.append( [ 0.5, radius ] )
    else:
        radius = l / ( 2 * ( n - 1 ) )
        mass = l**2 / n**2
        for y in np.linspace( radius, l - radius, n ):
            for x in np.linspace( 0.5 - l / 2 + radius, 0.5 + l / 2 - radius, n ):
                nx = x + 0.0 * radius * ( np.random.rand() - 0.5 )
                ny = y + 0.0 * radius * ( np.random.rand() - 0.5 )
                positions.append([nx, ny])
    positions = np.array(positions)
    nb_diracs = positions.shape[ 0 ]
    # dim = positions.shape[ 1 ]

    # OptimalTransport
    ot = OptimalTransport( domain, RadialFuncInBall() )
    ot.set_weights( np.ones( nb_diracs ) * radius**2 )
    ot.set_masses( np.ones( nb_diracs ) * mass )
    ot.set_positions( positions )
    ot.max_iter = 500
    ot.update_weights()

    ot.display_vtk( base_filename + "0.vtk", points=True, centroids=True )

    # history of centroids
    ce = ot.get_centroids()
    ce[:, 1] += radius / 5
    bh = [ce]

    dt = 1.0
    for num_iter in range( 15 ):
        print( "num_iter", num_iter )

        bh.append( ot.get_centroids() )
        fit_positions( ot, bh, dt )

        # display
        n1 = int( num_iter / 1 ) + 1
        ot.display_vtk( base_filename + "{}.vtk".format( n1 ), points=True, centroids=True )


os.system( "rm results/hd_*" )
run( 3, "results/hd_" )
