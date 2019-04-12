from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from pysdot import OptimalTransport
from util.fast_marching import GradGrid
import numpy as np


def in_box(Y, bbmin, bbmax):
    return np.logical_and(np.logical_and(Y[:,0] >= bbmin[0],
                                         Y[:,1] >= bbmin[1]),
                          np.logical_and(Y[:,0] >= bbmin[0],
                                         Y[:,1] >= bbmin[1]))

# constants
for na in [80,]: #[ 20, 40, 80, 160]:
    directory = "results/bimodal_crowd_{}".format( na )

    # constants
    alpha = 2*np.sqrt(1/np.pi)
    target_radius = 0.4* alpha / na
    timestep = target_radius/2
    epsilon = target_radius
    
    # positions
    t = np.linspace( 0 + target_radius, alpha - target_radius, na )
    x, y = np.meshgrid( t, t )
    positions = np.hstack( ( x.reshape( ( -1, 1 ) ), y.reshape( ( -1, 1 ) ) ) )

    # domain
    domain = ConvexPolyhedraAssembly()
    bb1min = [ 0, 0 ]
    bb1max = [ alpha, alpha ]
    bb2min = [ alpha, alpha/3 ]
    bb2max = [ 4*alpha/3, 2*alpha/3 ]
    bb3min = [ 4*alpha/3, 0 ]
    bb3max = [ 7*alpha/3, alpha ]
    domain.add_box( bb1min, bb1max )
    domain.add_box( bb2min, bb2max )
    domain.add_box( bb3min, bb3max )
    domain.display_boundaries_vtk( directory+"/bounds.vtk" )

    all_in_boxes = lambda Y: np.all(np.logical_or(in_box(bb1min, bb1max),
                                                  np.logical_or(in_box(Y, bb2min, bb2max),
                                                                in_box(Y, bb3min, bb3max))))

    #"draw((0,0)--(3,0)--(3,1)--
    #(4,1)--(4,0)--(7,0)--(7,3)--
    #(4,3)--(4,2)--(3,2)--(3,3)--(0,3)--cycle);\n
    domain_asy = ("draw((%g,%g)--(%g,%g)--(%g,%g)--"
                  "(%g,%g)--(%g,%g)--(%g,%g)--(%g,%g)--"
                  "(%g,%g)--(%g,%g)--(%g,%g)--(%g,%g)--(%g,%g)--cycle);\n" %
                  (0, 0, alpha, 0, alpha, alpha/3,
                   4*alpha/3, alpha/3, 4*alpha/3, 0, 7*alpha/3, 0, 7*alpha/3, alpha,
                   4*alpha/3, alpha, 4*alpha/3, 2*alpha/3, alpha, 2*alpha/3, alpha, alpha, 0, alpha))
    s = target_radius
    g = GradGrid( domain, [ [ 7*alpha/3-4*s, alpha-4*s ], [ 7*alpha/3-4*s, 0+4*s ] ], s )

    # iterations
    weights = np.ones( positions.shape[ 0 ] ) * target_radius ** 2
    timeout = np.zeros( positions.shape[ 0 ] )

    color_values = positions[ :, 1 ]
    color_values = ( color_values - np.min( color_values ) ) / np.ptp( color_values )
    
    h_weights = []
    h_positions = []

    T = 6.5
    nb_timesteps = int( T / timestep )
    display_timesteps = np.linspace(0, nb_timesteps-1, 100, dtype = int)
    
    ot = OptimalTransport(domain, RadialFuncInBall())
    ot.set_weights( weights )
    ot.set_masses( np.ones( positions.shape[ 0 ] ) * np.pi * target_radius ** 2 )
    for i in range( nb_timesteps ):
        print("iteration %d/%d for na=%d" % (i, nb_timesteps,na))

        # optimal weights
        ot.set_positions( positions )
        ot.adjust_weights()

        centroids = ot.get_centroids()

        # display
        if i in display_timesteps:
            print(i)
            j = display_timesteps.tolist().index(i)
            ot.display_asy( directory + "/pd_{:03}.asy".format( j ), values=color_values, linewidth=0.0005, dotwidth=target_radius * 0, closing=domain_asy, avoid_bounds=True )
            ot.display_vtk( directory + "/pd_{:03}.vtk".format( j ) )
            h_positions.append( positions )
            h_weights.append( weights )
    
        # update positions
        descent_direction = np.zeros_like(positions)
        for n in range( positions.shape[ 0 ] ):
            descent_direction[n,:] = g.grad( positions[ n, : ], 4 * target_radius )

        positions += timestep*(descent_direction + (centroids-positions)/epsilon)
        

        for n in range( positions.shape[ 0 ] ):
            if timeout[ n ] == 0 and positions[ n, 0 ] > 4*alpha/3:
                timeout[ n ] = i + 1

    # output with timeout information
    timeout = ( timeout - np.min( timeout ) ) / np.ptp( timeout )
    for i in range( len( h_weights ) ):
        ot.display_asy( directory + "/pd_timeout_{:03}.asy".format( i ), "in_ball(weight**0.5)", h_positions[ i ], h_weights[ i ], domain, values = timeout, linewidth = 0.005, dotwidth = target_radius * 0, closing = domain_asy, avoid_bounds = True )
