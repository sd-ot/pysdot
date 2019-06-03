from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.domain_types import ScaledImage
from pysdot import OptimalTransport
from pysdot import PowerDiagram
from skimage.transform import downscale_local_mean
import pylab as plt
import numpy as np
import imageio

t = np.linspace(-1,1,100)
x, y = np.meshgrid(t,t)
img = np.exp( -4 * (x**2 + y**2) )
img /= np.mean(img)

# domain
domain = ScaledImage([0, 0], [1, 1], img)

# diracs
def quantization(ot, tau=.3, niter=10):
    for _ in range(niter):
        ot.adjust_weights()
        B = ot.get_centroids()
        ot.set_positions( ot.get_positions() + tau * ( B - ot.get_positions() ) )
    ot.adjust_weights()

# solve
n = 40
positions = []
for y in range( n ):
    for x in range( n ):
        positions.append( [ ( y + 0.25 + 0.5 * np.random.rand() ) / n, ( x + 0.25 + 0.5 * np.random.rand() ) / n ] )
ot = OptimalTransport(np.array(positions), None, domain)
quantization(ot, 0.1, 10)

# display
# ot.pd.display_vtk( "results/pd.vtk", centroids=True )

# power diagram with a simple [0,1]^2 domain
pd = PowerDiagram(ot.get_positions(), ot.get_weights())
img = pd.image_integrals( [ 0, 0 ], [ 1, 1 ], [ 100, 100 ] )

for d in range( 2 ):
    plt.subplot( 1, 2, d + 1 )
    plt.imshow( img[ :, :, d ] / img[ :, :, 2 ] )
    plt.colorbar()

plt.show()

plt.plot( img[ 50, :, 0 ] / img[ 50, :, 2 ] )
plt.show()
