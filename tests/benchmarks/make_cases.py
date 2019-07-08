from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.domain_types import ScaledImage
from pysdot import OptimalTransport
from pysdot import PowerDiagram
import pylab as plt
import numpy as np
import itertools


def make_grid(nb_diracs, dim, rand_val=0):
    n = int( np.sqrt( nb_diracs ) )
    items = [ range( n ) for i in range( dim ) ]
    positions = []
    for i in itertools.product( *items ):
        positions.append( i )
    return ( np.array(positions) + ( 1 - rand_val ) / 2 + rand_val * np.random.rand( n**dim, dim ) ) / n


def make_case( name, nb_diracs, dim ):
    # default domain
    if name == "random":
        positions = np.random.rand( nb_diracs, dim )
    elif name == "grid":
        positions = make_grid( nb_diracs, dim )
    elif name == "grid_with_rand":
        positions = make_grid( nb_diracs, dim, rand_val=1 )
    elif name == "faces":
        # voronoi with 100 points
        pd = PowerDiagram( np.random.rand( 5, dim ) )

        # quantization
        lot = OptimalTransport( positions = make_grid( nb_diracs, dim ) )
        lot.obj_max_dw = 1e-5
        lot.verbosity = 1
        for ratio in [ 1 - 0.85**n for n in range( 50 ) ]:
            # density
            img_size = 1000
            img_points = []
            items = [ range( img_size ) for i in range( dim ) ]
            for i in itertools.product( *items ):
                img_points.append( i )
            img = pd.distances_from_boundaries( np.array( img_points ) / img_size ).reshape( ( img_size, img_size ) )
            img = ( 1 - ratio ) + ratio * np.exp( - ( 100 * img )**2 )
            lot.set_domain( ScaledImage( [ 0, 0 ], [ 1, 1 ], img / np.mean( img ) ) )

            # opt
            for _ in range( 10 ):
                lot.adjust_weights()
                B = lot.get_centroids()
                lot.set_positions( lot.get_positions() + 0.3 * ( B - lot.get_positions() ) )

        positions = lot.get_positions()
        plt.plot( positions[ :, 0 ], positions[ :, 1 ], "." )
        plt.show()

    np.save( "/data/{}_n{}_d{}_voro.npy".format( name, nb_diracs, dim ), ( positions[ :, 0 ], positions[ :, 1 ] ) )

    # solve
    if nb_diracs < 32000000:
        ot = OptimalTransport(positions)
        # ot.verbosity = 1

        # solve
        ot.adjust_weights()

        # display
        # ot.display_vtk( "results/pd.vtk" )
        np.save( "/data/{}_n{}_d{}.npy".format( name, nb_diracs, dim ), ( positions[ :, 0 ], positions[ :, 1 ], ot.get_weights() ) )


ns = [ int( 1000000 * n ) for n in [ 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128 ] ]
gs = [ "grid_with_rand" ] # [ "random", "grid", "grid_with_rand" ]

for t in gs:
    print( "t=", t )
    for n in ns:
        print( "n=",n )
        make_case( t, n, 2 )
