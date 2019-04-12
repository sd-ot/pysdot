from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncEntropy
from pysdot import OptimalTransport
import numpy as np

# constants
for n in [ 10 ]:
    directory = "results/diffusion_{}".format( n )

    # constants
    eps = n ** -0.5

    # domain
    domain = ConvexPolyhedraAssembly()
    domain.add_convex_polyhedron( [
         [ -1, -1, -1,  0 ],
         [ -1, -1,  0, -1 ],
         [ +2, -1,  1, -1 ],
         [ -1, +2,  0, +1 ]
    ] )

    domain.display_boundaries_vtk( directory + "/bounds.vtk" )

    # 
    positions = []
    for y in np.linspace( 0, 1, n ):
        for x in np.linspace( 0, 1, n ):
            positions.append( [ x, y ] )
    positions = np.array( positions )

    # iterations
    max_w = -1
    min_w = 0

    ot = OptimalTransport(domain, RadialFuncEntropy( eps ))
    ot.set_masses( np.ones( positions.shape[ 0 ] ) )

    weights = np.zeros( positions.shape[ 0 ] )
    for i in range( 101 ):
        # optimal weights
        ot.set_positions( positions )
        ot.adjust_weights()

        weights = ot.get_weights()
        min_w = np.min( weights )
        max_w = np.max( weights )

        # display
        if i % 10 == 0:
            ot.display_asy( directory + "/we_{:03}.asy".format( int( i / 10 ) ), values = ( weights - min_w ) / ( max_w - min_w ), linewidth=0.002 )
            ot.display_asy( directory + "/pd_{:03}.asy".format( int( i / 10 ) ), values = ( weights - min_w ) / ( max_w - min_w ), linewidth=0.002, min_rf=0, max_rf=0.35 )
        ot.display_vtk( directory + "/pd_{:03}.vtk".format( i ) )
    
        # update positions
        d = 0.75
        positions = d * ot.get_centroids() + ( 1 - d ) * positions

