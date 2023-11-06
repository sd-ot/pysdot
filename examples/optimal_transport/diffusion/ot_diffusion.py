from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncEntropy
from pysdot import OptimalTransport
import numpy as np

# constants
for n in [ 80 ]:
    directory = "results/diffusion_{}".format( n )

    # constants
    h = 1./n
    eps = np.sqrt(h)
    tau = h/10
    T = 2.5

    # domain
    domain = ConvexPolyhedraAssembly()
    domain.add_convex_polyhedron( [
         [ -1, -1, -1,  0 ],
         [ -1, -1,  0, -1 ],
         [ +2, -1,  1, -1 ],
         [ -1, +2,  0, +1 ]
    ] )

    domain.display_boundaries_vtk( directory + "/bounds.vtk" )

    positions = []
    for y in np.linspace( 0, 1, n ):
        for x in np.linspace( 0, 1, n ):
            if (x-.5)**2 + (y-.5)**2 <= .5**2:
                positions.append( [ x, y ] )
    positions = np.array( positions )
    N = positions.shape[0]
    
    # iterations

    ot = OptimalTransport(domain, RadialFuncEntropy( eps ))
    ot2 = OptimalTransport(domain)
    ot.set_masses( np.ones( positions.shape[ 0 ] )/N )

    weights = np.zeros( positions.shape[ 0 ] )
    niter =  int(T/tau)
    display_timesteps = np.linspace(0, niter-1, 6, dtype = int)
    save_timesteps = np.linspace(0, niter-1, 100, dtype = int)
    max_rho = -1

    h_positions = []
    for i in range(niter):
        #print("iteration %d/%d" % (i,niter))
        # optimal weights
        ot.set_positions( positions )
        ot.adjust_weights()

        weights = ot.get_weights()
        
        # display
        if i in display_timesteps:
            print(i)
            j = display_timesteps.tolist().index(i)
            diffpos = ot.get_centroids() - positions
            rho = np.exp(-(diffpos[:,0]**2 + diffpos[:,1]**2 - weights)/eps)
            #if max_rho<0:
            max_rho = max(rho)
            ot.display_asy( directory + "/pd_{:03}.asy".format( int( j ) ), values = rho / max_rho, linewidth=0.00002)
        #ot.display_vtk( directory + "/pd_{:03}.vtk".format( i ) )    
        if i in save_timesteps:
            h_positions.append( positions.copy() )
    
        # update positions
        positions =  positions + (tau/eps) * (ot.get_centroids() - positions)

    np.save(directory + "/positions.npy", h_positions)

