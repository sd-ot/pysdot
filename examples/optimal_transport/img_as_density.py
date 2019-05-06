from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import OptimalTransport
from skimage.transform import downscale_local_mean
import pylab as plt
import numpy as np
import imageio

dws = 5
img = imageio.imread("clay.jpg")
img = img[ :, :, 1 ]

beg = int( img.shape[ 1 ] / 2 - img.shape[ 0 ] / 2 )
end = beg + img.shape[ 0 ]
img = img[ :, beg : end ]
img = downscale_local_mean(img, (dws, dws))
img = np.max( img ) * 1.05 - img
img /= np.sum( img )
# plt.imshow( img )
# plt.show()

# domain
domain = ConvexPolyhedraAssembly()
domain.add_img([0, 0], [1, 1], img)

# diracs
nd = 100000
ot = OptimalTransport(domain)
ot.set_positions(np.random.rand(nd, 2))
ot.set_weights(np.ones(nd)/nd)
ot.obj_max_dw = 1e-5
ot.verbosity = True

# solve
ot.adjust_weights( relax=1.0 )

# display
ot.pd.display_vtk( "results/pd.vtk", centroids=True )

