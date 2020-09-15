from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from pysdot import OptimalTransport
from pysdot import PowerDiagram
import pylab as plt
import numpy as np
import unittest

domain = ConvexPolyhedraAssembly()
domain.add_box([0, 0], [1, 1])

positions = [ [ 0.25, 0.5 ], [ 0.75, 0.5 ] ]
weights = [ 0.25**2, 0.25**2 ]

pd = PowerDiagram( positions, weights, domain, radial_func=RadialFuncInBall())

img = pd.image_integrals( [ 0, 0 ], [ 1, 1 ], [ 100, 100 ] )

img[ :, :, 0 ] *= 1e4
img[ :, :, 1 ] *= 1e4

# for d in range( 3 ):
#     plt.subplot( 1, 3, d + 1 )
#     plt.imshow( img[ :, :, d ] )
#     plt.colorbar()

# plt.show()

# w = 0.1

# print( np.pi * w )
# print( np.pi / 2 * w**2 )

# for x in [ 0.5, 0.0 ]:

#     pd.set_weights([w])

#     areas = pd.integrals()
#     smoms = pd.second_order_moments()
#     print( areas )
#     print( smoms )

# class TestInBall_2D(unittest.TestCase):
#     def setUp(self):
#         self.domain = ConvexPolyhedraAssembly()
#         self.domain.add_box([0, 0], [1, 1])

#     def test_unit(self):
#         pd = PowerDiagram(self.domain)
#         pd.set_positions([[0.0, 0.0]])
#         pd.set_weights([0.0])

#         areas = pd.integrals()
#         print( areas )
#         # self.assertAlmostEqual(areas[0], 1.0)

#         # centroids = pd.centroids()
#         # self.assertAlmostEqual(centroids[0][0], 0.5)
#         # self.assertAlmostEqual(centroids[0][1], 0.5)

