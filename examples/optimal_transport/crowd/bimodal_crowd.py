from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from pysdot import OptimalTransport
from util.fast_marching import GradGrid
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
    domain = ConvexPolyhedraAssembly()
    domain.add_box( [ 0, 0 ], [ 3, 3 ] )
    domain.add_box( [ 3, 1 ], [ 4, 2 ] )
    domain.add_box( [ 4, 0 ], [ 7, 3 ] )
    domain.display_boundaries_vtk( directory + "/bound.vtk" )

    domain_asy = "draw((0,0)--(3,0)--(3,1)--(4,1)--(4,0)--(7,0)--(7,3)--(4,3)--(4,2)--(3,2)--(3,3)--(0,3)--cycle);\n"

    s = 0.02
    g = GradGrid( domain, [ [ 7-s, 3-s ], [ 7-s, 0+s ] ], s )

    # iterations
    ot = OptimalTransport(domain, RadialFuncInBall())
    ot.set_weights( np.ones( positions.shape[ 0 ] ) * target_radius ** 2 )
    ot.set_masses( np.ones( positions.shape[ 0 ] ) * np.pi * target_radius ** 2 )

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
        ot.set_positions( positions )
        ot.adjust_weights()

        # display
        d = 2 * na
        if i % d == 0:
            ot.pd.display_asy( directory + "/pd_{:03}.asy".format( int( i / d ) ), values = color_values, linewidth = 0.005, dotwidth = target_radius * 0, closing = domain_asy, avoid_bounds = True )
            ot.display_vtk( directory + "/pd_{:03}.vtk".format( int( i / d ) ) )
            h_positions.append( positions )
            h_weights.append( ot.get_weights() )
    
        # update positions
        positions = ot.get_centroids()

        for n in range( positions.shape[ 0 ] ):
            if timeout[ n ] == 0 and positions[ n, 0 ] > 4:
                timeout[ n ] = i + 1

    # output with timeout information
    timeout = ( timeout - np.min( timeout ) ) / np.ptp( timeout )
    for i in range( len( h_weights ) ):
        ot.set_positions( h_positions[ i ] )
        ot.set_weights( h_weights[ i ] )
        ot.pd.display_asy( directory + "/pd_timeout_{:03}.asy".format( i ), "in_ball(weight**0.5)", values = timeout, linewidth = 0.005, dotwidth = target_radius * 0, closing = domain_asy, avoid_bounds = True )
