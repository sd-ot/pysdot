from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncPpWmR2
from pysdot import PowerDiagram
import numpy as np
import unittest


class TestPpWmR2_2D(unittest.TestCase):
    def setUp(self):
        self.domain = ConvexPolyhedraAssembly()
        self.domain.add_box([0.0, 0.0], [10.0, 10.0])

    def test_integral(self):
        # diracs
        rd = 2.0
        pd = PowerDiagram(domain=self.domain, radial_func=RadialFuncPpWmR2())
        pd.set_positions(np.array([[1.0, 0.0], [5.0, 5.0]]))
        pd.set_weights(np.array([rd**2, rd**2]))

        # integrals
        ig = pd.integrals()
        print( ig )
        # self.assertAlmostEqual(ig[ 0 ], 2 * np.pi)
        # self.assertAlmostEqual(ig[ 1 ], 8 * np.pi)

        # centroids
        ct = pd.centroids()
        print( ct )

        # derivatives





if __name__ == '__main__':
    unittest.main()
