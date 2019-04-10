import py_power_diagram_test_context
import py_power_diagram as pd
import fast_marching as fm
import numpy as np

# constants
for na in [ 20, 40, 80, 160, 200 ]: # 
    directory = "results/bimodal_crowd_{}".format( na )

    # constants
    target_radius = 3 * 0.45 / na

    # positions
    t = np.linspace( 0 + target_radius, 3 - target_radius, na )
    x, y = np.meshgrid( t, t )
    positions = np.hstack( ( x.reshape( ( -1, 1 ) ), y.reshape( ( -1, 1 ) ) ) )

    # domain
    domain = pd.domain_types.ConvexPolyhedraAssembly()
    domain.add_box( [ 0, 0 ], [ 3, 3 ], 1 / ( np.pi * target_radius ** 2 ) )
    domain.add_box( [ 3, 1 ], [ 4, 2 ], 1 / ( np.pi * target_radius ** 2 ) )
    domain.add_box( [ 4, 0 ], [ 7, 3 ], 1 / ( np.pi * target_radius ** 2 ) )
    domain.display_boundaries_vtk( directory + "/bound.vtk" )

    domain_asy = "draw((0,0)--(3,0)--(3,1)--(4,1)--(4,0)--(7,0)--(7,3)--(4,3)--(4,2)--(3,2)--(3,3)--(0,3)--cycle);\n"

    s = 0.02
    g = fm.GradGrid( domain, [ [ 7-s, 3-s ], [ 7-s, 0+s ] ], s )

    # iterations
    weights = np.ones( positions.shape[ 0 ] ) * target_radius ** 2
    timeout = np.zeros( positions.shape[ 0 ] )

    color_values = positions[ :, 1 ]
    color_values = ( color_values - np.min( color_values ) ) / np.ptp( color_values )

    h_weights = []
    h_positions = []
    nb_timesteps = int( 22 / target_radius )
    for i in range( nb_timesteps ):
        # change positions
        for n in range( positions.shape[ 0 ] ):
            positions[ n, : ] += 0.5 * target_radius * g.grad( positions[ n, : ] )

        # optimal weights
        weights = pd.optimal_transport_2( "in_ball(weight**0.5)", positions, weights, domain )

        # display
        d = 2 * na
        if i % d == 0:
            pd.display_asy( directory + "/pd_{:03}.asy".format( int( i / d ) ), "in_ball(weight**0.5)", positions, weights, domain, values = color_values, linewidth = 0.005, dotwidth = target_radius * 0, closing = domain_asy, avoid_bounds = True )
            pd.display_vtk( directory + "/pd_{:03}.vtk".format( int( i / d ) ), "in_ball(weight**0.5)", positions, weights, domain )
            h_positions.append( positions )
            h_weights.append( weights )
    
        # update positions
        positions = pd.get_centroids( "in_ball(weight**0.5)", positions, weights, domain )

        for n in range( positions.shape[ 0 ] ):
            if timeout[ n ] == 0 and positions[ n, 0 ] > 4:
                timeout[ n ] = i + 1

    # output with timeout information
    timeout = ( timeout - np.min( timeout ) ) / np.ptp( timeout )
    for i in range( len( h_weights ) ):
        pd.display_asy( directory + "/pd_timeout_{:03}.asy".format( i ), "in_ball(weight**0.5)", h_positions[ i ], h_weights[ i ], domain, values = timeout, linewidth = 0.005, dotwidth = target_radius * 0, closing = domain_asy, avoid_bounds = True )
