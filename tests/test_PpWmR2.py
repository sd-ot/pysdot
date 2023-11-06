from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncPpWmR2
from pysdot.solvers import Scipy
from pysdot import PowerDiagram
import numpy as np
import unittest


class TestPpWmR2_2D(unittest.TestCase):
    def setUp(self):
        self.domain = ConvexPolyhedraAssembly()
        self.domain.add_box([0.0, 0.0], [10.0, 10.0])

        self.solver = Scipy.Solver()

    def test_integrals(self):
        # diracs
        rd = 2.0
        pd = PowerDiagram(domain=self.domain, radial_func=RadialFuncPpWmR2())
        pd.set_positions(np.array([[1.0, 0.0], [5.0, 5.0]]))
        pd.set_weights(np.array([rd**2, rd**2]))

        # integrals
        ig = pd.integrals()

        # Integrate[ Integrate[ ( 4 - ( x * x + y * y ) ) * UnitStep[ 2^2 - x^2 - y^2 ] , { x, -1, 2 } ], { y, 0, 3 } ]
        self.assertAlmostEqual(ig[ 0 ], 10.97565662)
        self.assertAlmostEqual(ig[ 1 ], 8 * np.pi) 

        # centroids
        ct = pd.centroids()

        # Integrate[ Integrate[ x * ( 4 - ( x * x + y * y ) ) * UnitStep[ 2^2 - x^2 - y^2 ] , { x, -1, 2 } ], { y, 0, 3 } ]
        self.assertAlmostEqual(ct[ 0 ][ 0 ], 1.18937008)
        self.assertAlmostEqual(ct[ 0 ][ 1 ], 0.69699702) 

        self.assertAlmostEqual(ct[ 1 ][ 0 ], 5)
        self.assertAlmostEqual(ct[ 1 ][ 1 ], 5)

    def test_derivatives(self):
        # diracs
        rd = 2.0
        weights = np.array([rd**2, rd**2])
        pd = PowerDiagram(domain=self.domain, radial_func=RadialFuncPpWmR2())
        pd.set_positions(np.array([[1.0, 0.0], [5.0, 5.0]]))
        pd.set_weights(weights.copy())

        # derivatives
        num = pd.der_integrals_wrt_weights()

        ndi = pd.weights.shape[0]
        mat = np.zeros((ndi, ndi))
        for i in range(ndi):
            for o in range(num.m_offsets[i + 0], num.m_offsets[i + 1]):
                mat[i, num.m_columns[o]] = num.m_values[o]

        # numerical
        eps = 1e-6
        res = pd.integrals()
        delta = np.max(np.abs(mat)) * 100 * eps
        for i in range(ndi):
            pd.set_weights(np.array(
                [weights[j] + eps * (i == j) for j in range(ndi)]
            ))
            des = pd.integrals()
            der = (des - res) / eps
            for j in range(ndi):
                #self.assertAlmostEqual(mat[i, j], der[j], delta=delta)
                print( mat[i, j], der[j] )


if __name__ == '__main__':
    unittest.main()
