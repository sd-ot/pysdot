from pysdot.domain_types import ScaledImage
from pysdot import OptimalTransport
import pylab as plt
import numpy as np

# helper function
def quantization(ot, tau=.3, niter=10):
    for _ in range(niter):
        ot.adjust_weights()
        B = ot.get_centroids()
        ot.set_positions( ot.get_positions() + tau * ( B - ot.get_positions() ) )
    ot.adjust_weights()

# initial positions
n = 30
positions = []
for y in range( n ):
    for x in range( n ):
        positions.append( [ np.random.rand(), np.random.rand() ] )
ot = OptimalTransport(np.array(positions))
ot.verbosity = 1

# solve
for l in [ 0.5, 1, 2, 4 ]: # , 8, 16
    print( l )
    tx = np.linspace(-1,1,500)
    ty = np.linspace(-1,1,500)
    x, y = np.meshgrid(tx,ty)
    img = np.exp( -l * (x**2 + y**2) )
    img /= np.mean(img)
    
    # domain
    #ot.set_domain(ScaledImage([0, 0], [1, 1], img))
    #quantization(ot, 0.1, 10)

# display
#plt.plot( ot.pd.positions[ :, 0 ], ot.pd.positions[ :, 1 ], '.' )

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, img, rstride=20, cstride=20)

plt.show()
