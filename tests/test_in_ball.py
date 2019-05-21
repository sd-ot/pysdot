from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from pysdot import PowerDiagram
import numpy as np
import unittest

domain = ConvexPolyhedraAssembly()
domain.add_box([0, 0], [1, 1])

w = 0.1

print( np.pi * w )
print( np.pi / 2 * w**2 )

for x in [ 0.5, 0.0 ]:
    pd = PowerDiagram(domain,radial_func=RadialFuncInBall())
    pd.set_positions([[x, 0.5]])

    pd.set_weights([w])

    areas = pd.integrals()
    smoms = pd.second_order_moments()
    print( areas )
    print( smoms )

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

