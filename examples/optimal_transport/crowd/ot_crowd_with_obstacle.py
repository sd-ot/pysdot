from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from pysdot import OptimalTransport
from util.fast_marching import GradGrid
import numpy as np

times = []
for obstacle_radius in [ 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35 ]:
    for na in [ 50 ]: # 
        directory = "results/crowd_with_obstacle_{}_w{}".format( na, int( 100 * obstacle_radius ) )

        # constants
        target_radius = 0.45 / na
        opened_width = 0.2
        pos_door = 2

        # positions
        t = np.linspace( 0 + target_radius, 1 - target_radius, na )
        x, y = np.meshgrid( t, t )
        positions = np.hstack( ( x.reshape( ( -1, 1 ) ), y.reshape( ( -1, 1 ) ) ) )

        limx = pos_door + 1.2 / opened_width

        #
        pon = []
        for i in range( 16 ):
            a = 2 * np.pi * i / 16
            pon.append( [ 1.5 + obstacle_radius * np.cos( a ), 0.5 + obstacle_radius * np.sin( a ), np.cos( a ), np.sin( a ) ] )

        # domain
        domain = ConvexPolyhedraAssembly()
        domain.add_box( [ 0, 0 ], [ 2, 1 ] )
        domain.add_box( [ pos_door, 0.5 - opened_width / 2 ], [ limx, 0.5 + opened_width / 2 ] )
        if obstacle_radius:
            domain.add_convex_polyhedron( np.array( pon ), -1.0 )
        domain.display_boundaries_vtk( directory + "/bound.vtk" )

        # domain_asy = "draw((0,0)--(3,0)--(3,1)--(4,1)--(4,0)--(7,0)--(7,3)--(4,3)--(4,2)--(3,2)--(3,3)--(0,3)--cycle);\n"

        s = 0.5 * target_radius
        g = GradGrid( domain, [ [ limx - 2 * s, 0.5 ] ], s )

        # iterations
        color_values = positions[ :, 1 ]
        color_values = ( color_values - np.min( color_values ) ) / np.ptp( color_values )

        ot = OptimalTransport(domain, RadialFuncInBall())
        ot.set_weights( np.ones( positions.shape[ 0 ] ) * target_radius ** 2 )
        ot.set_masses( np.ones( positions.shape[ 0 ] ) * np.pi * target_radius ** 2 )

        nb_timesteps = int( 20 / target_radius )
        for i in range( nb_timesteps ):
            # change positions
            weights = ot.get_weights()
            for n in range( positions.shape[ 0 ] ):
                c = 1 / ( 10 * weights[ n ] / 7e-4 + 1 )
                positions[ n, : ] += c * 0.5 * target_radius * g.grad( positions[ n, : ] )

            # optimal weights
            ot.set_positions( positions )
            ot.adjust_weights()

            # display
            d = 5
            if i % d == 0:
                ot.display_vtk( directory + "/pd_{:03}.vtk".format( int( i / d ) ) )
        
            # update positions
            positions = ot.get_centroids()

            if np.min( positions[ :, 0 ] ) > pos_door:
                times.append( i )
                break

print( times )
