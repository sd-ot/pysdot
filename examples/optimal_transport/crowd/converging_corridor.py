from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from pysdot import OptimalTransport
import numpy as np

# constants
for n in [ 160 ]: # 20, 40, 80, 
    directory = "results/converging_corridor_{}".format( n )

    t = np.linspace(-2,2,n)
    h = 2./n
    x,y = np.meshgrid(t,t)
    positions = np.hstack((np.reshape(x,(n*n,1)),
                           np.reshape(y,(n*n,1))))
    R2 = positions[ :, 0 ]**2 + positions[ :, 1 ]**2
    positions = positions[ R2 <= 4 ]
    positions = positions[   positions[:,0] + positions[:,1] > 1.0 / n ]
    positions = positions[ - positions[:,0] + positions[:,1] > 1.0 / n ]
    N = positions.shape[ 0 ]
    rho0 = 1 / np.pi
    mass = 0.25 * rho0 * np.pi * 4 / N
    target_radius = ( mass / np.pi )**0.5

    # iterations
    weights = np.ones( positions.shape[ 0 ] ) * target_radius**2

    domain = ConvexPolyhedraAssembly()
    domain.add_convex_polyhedron([
        [  0, 0, +1, -1 ],
        [  9, 9,  0, +1 ],
        [ -9, 9, -1, -1 ],
    ])
    domain.display_boundaries_vtk( directory + "/bounds.vtk" )

    color_values = 0.5 * np.linalg.norm( positions, axis=1, keepdims=True, ord=2 )
    color_values = ( color_values - np.min( color_values ) ) / ( np.max( color_values ) - np.min( color_values ) )

    ot = OptimalTransport(domain, RadialFuncInBall())
    ot.set_weights( weights )
    ot.set_masses( np.ones( positions.shape[ 0 ] ) * mass )

    nb_timesteps = int( 3 / target_radius )
    for i in range( nb_timesteps ):
        # change positions
        positions -= 0.4 * target_radius / np.linalg.norm( positions, axis=1, keepdims=True, ord=2 ) * positions
        ot.set_positions(positions)
    
        # optimal weights
        ot.adjust_weights()

        # display
        d = int( n / 5 )
        if i % d == 0:
            # ot.display_asy( directory + "/pd_{:03}.asy".format( int( i / d ) ), "in_ball(weight**0.5)", positions, weights, domain, values = color_values, linewidth=0.002 )
            ot.display_vtk( directory + "/pd_{:03}.vtk".format( int( i / d ) ) )
    
        # update positions
        positions = ot.get_centroids()

