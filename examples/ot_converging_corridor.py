import py_power_diagram_test_context
import py_power_diagram as pd
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

    cp = [
         0, 0, +1, -1,
         9, 9,  0, +1,
        -9, 9, -1, -1,
    ]

    # iterations
    weights = np.ones( positions.shape[ 0 ] ) * target_radius**2

    domain = pd.domain_types.ConvexPolyhedraAssembly()
    domain.add_convex_polyhedron( cp, 1 / mass )
    domain.display_boundaries_vtk( directory + "/bounds.vtk" )

    color_values = 0.5 * np.linalg.norm( positions, axis=1, keepdims=True, ord=2 )
    color_values = ( color_values - np.min( color_values ) ) / ( np.max( color_values ) - np.min( color_values ) )

    nb_timesteps = int( 3 / target_radius )
    for i in range( nb_timesteps ):
        # change positions
        positions -= 0.4 * target_radius / np.linalg.norm( positions, axis=1, keepdims=True, ord=2 ) * positions

        # optimal weights
        weights = pd.optimal_transport_2( "in_ball(weight**0.5)", positions, weights, domain )

        # display
        d = int( n / 5 )
        if i % d == 0:
            pd.display_asy( directory + "/pd_{:03}.asy".format( int( i / d ) ), "in_ball(weight**0.5)", positions, weights, domain, values = color_values, linewidth=0.002 )
            pd.display_vtk( directory + "/pd_{:03}.vtk".format( int( i / d ) ), "in_ball(weight**0.5)", positions, weights, domain )
    
        # update positions
        positions = pd.get_centroids( "in_ball(weight**0.5)", positions, weights, domain )

