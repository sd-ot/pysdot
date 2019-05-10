from pysdot.domain_types import ScaledImage
from pysdot import OptimalTransport
from skimage.transform import downscale_local_mean
import pylab as plt
import numpy as np
import imageio

# dws = 5
# # img = imageio.imread("clay.jpg")
# # img = img[ :, :, 1 ]

# beg = int( img.shape[ 1 ] / 2 - img.shape[ 0 ] / 2 )
# end = beg + img.shape[ 0 ]
# img = img[ :, beg : end ]
# img = downscale_local_mean(img, (dws, dws))
# img = np.max( img ) * 1.05 - img
# img /= np.sum( img )
# plt.imshow( img )
# plt.show()

t = np.linspace(-1,1,10)
x, y = np.meshgrid(t,t)
img = np.exp( -2 * (x**2 + y**2) )
img /= np.mean(img)

# domain
domain = ScaledImage([0, 0], [1, 1], img)

# diracs
def quantization(ot, tau=.1, niter=100):
    for _ in range(niter):
        ot.adjust_weights(relax=1.0, initial_weights=np.zeros(ot.nb_diracs()))
        B = ot.get_centroids()
        ot.set_positions( ot.get_positions() + tau * ( B - ot.get_positions() ) )

nd = 1000

ot = OptimalTransport()
ot.set_positions(np.random.rand(nd, 2))
ot.set_weights(np.ones(nd)/nd)
ot.set_domain(domain)
ot.obj_max_dw = 1e-5
ot.verbosity = True
ot.max_iter = 100

# solve
quantization(ot, 0.1, 20)

# display
ot.pd.display_vtk( "results/pd.vtk", centroids=True )

