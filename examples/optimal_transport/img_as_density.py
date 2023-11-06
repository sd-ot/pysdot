from pysdot.domain_types import ScaledImage
from pysdot import OptimalTransport
import pylab as plt
import numpy as np

# helper function
def quantization(ot, tau=.3, niter=10):
    for iter in range(niter):
        if ot.verbosity >= 2:
            print( "niter quant:", iter )
        ot.adjust_weights()
        B = ot.get_centroids()
        ot.set_positions( ot.get_positions() + tau * ( B - ot.get_positions() ) )
    ot.adjust_weights()

# initial positions
n = 40
positions = []
for y in range( n ):
    for x in range( n ):
        positions.append( [ ( y + 0.25 + 0.5 * np.random.rand() ) / n, ( x + 0.25 + 0.5 * np.random.rand() ) / n ] )
ot = OptimalTransport(np.array(positions))
ot.verbosity = 2

# solve
for l in [ 1, 2, 4, 8 ]:
    t = np.linspace(-1,1,100)
    x, y = np.meshgrid(t,t)
    img = np.exp( -l * (x**2 + y**2) )
    img /= np.mean(img)
    
    # domain
    ot.set_domain(ScaledImage([0, 0], [1, 1], img))
    quantization(ot, 0.1, 10)

# display
ot.pd.display_vtk( "results/pd.vtk", centroids=True )

# optimal transport with a simple [0,1]^2 domain
# ot = OptimalTransport(ot.get_positions())
# ot.adjust_weights()

# img = ot.pd.image_integrals( [ 0, 0 ], [ 1, 1 ], [ 100, 100 ] )

# for d in range( 2 ):
#     plt.subplot( 1, 2, d + 1 )
#     plt.imshow( img[ :, :, d ] / img[ :, :, 2 ] )
#     plt.colorbar()

# plt.show()

# plt.plot( img[ 50, :, 0 ] / img[ 50, :, 2 ], '+' )
# plt.show()
