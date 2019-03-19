import py_power_diagram_test_context
import py_power_diagram as pd
import numpy as np

# constants
for n in [ 10 ]:
    directory = "results/diffusion_{}".format( n )

    # constants
    eps = n ** -0.5
    rfu = "exp((w-r**2)/{:.16f})".format( eps )

    # domain
    domain = pd.domain_types.ConvexPolyhedraAssembly()
    domain.add_convex_polyhedron( [
         [ -1, -1, -1,  0 ],
         [ -1, -1,  0, -1 ],
         [ +2, -1,  1, -1 ],
         [ -1, +2,  0, +1 ]
    ] ) # target mass => 1/N

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
    weights = np.zeros( positions.shape[ 0 ] )
    for i in range( 101 ):
        # optimal weights
        weights = pd.optimal_transport_2( rfu, positions, weights, domain )

        min_w = np.min( weights )
        max_w = np.max( weights )

        # display
        if i % 10 == 0:
            pd.display_asy( directory + "/we_{:03}.asy".format( int( i / 10 ) ), rfu, positions, weights, domain, values = ( weights - min_w ) / ( max_w - min_w ), linewidth=0.002 )
            pd.display_asy( directory + "/pd_{:03}.asy".format( int( i / 10 ) ), rfu, positions, weights, domain, values = ( weights - min_w ) / ( max_w - min_w ), linewidth=0.002, min_rf=0, max_rf=0.35 )
        pd.display_vtk( directory + "/pd_{:03}.vtk".format( i ), rfu, positions, weights, domain )
    
        # update positions
        d = 0.75
        positions = d * pd.get_centroids( rfu, positions, weights, domain ) + ( 1 - d ) * positions

