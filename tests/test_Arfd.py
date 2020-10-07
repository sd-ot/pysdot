from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncArfd
from pysdot.solvers import Scipy
from pysdot import PowerDiagram
import numpy as np
import unittest

def check_ders( test, pd ):
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
    weights = pd.get_weights().copy()
    for i in range(ndi):
        pd.set_weights(np.array(
            [weights[j] + eps * (i == j) for j in range(ndi)]
        ))
        des = pd.integrals()
        der = (des - res) / eps
        for j in range(ndi):
            test.assertAlmostEqual(mat[i, j], der[j], places=5)


class TestArfd_2D(unittest.TestCase):
    def setUp(self):
        rf = RadialFuncArfd(
            lambda r: ( 1 - r * r ) * ( r < 1 ), # value
            lambda w: 1 / w ** 0.5, # input (radius) scaling
            lambda w: w, # output scaling
            lambda r: r < 1, # value for the der wrt weight
            lambda w: 1 / w ** 0.5, # input scaling for the der wrt weight
            lambda w: 1, # output scaling for the der wrt weight
            [ 1 ] # stops (radii value where we may have discontinuities)
        )

        # should use only 2 polynomials
        self.assertEqual(rf.nb_polynomials(), 2)

        # set up a domain, with only 1 dirac
        domain = ConvexPolyhedraAssembly()
        domain.add_box([0.0, 0.0], [2.0, 1.0])

        self.pd = PowerDiagram(domain=domain, radial_func=rf)

    def test_integrals(self):
        self.pd.set_positions(np.array([[0.0, 0.0]]))

        # weights and expected values (w = weight, i = integral, c = centroid)
        l = [
            { "w": 0.5, "i": np.pi / 32, "c": [ 8 * 2 ** 0.5 / ( 15 * np.pi ) ] * 2 },
            { "w":   1, "i": np.pi /  8, "c": [         16.0 / ( 15 * np.pi ) ] * 2 },
            { "w":  10, "i":    50 /  3, "c": [  23.0 /  25  ,  49.0 /  100   ]     },
            { "w": 100, "i":   590 /  3, "c": [ 293.0 / 295  , 589.0 / 1180   ]     },
        ]

        # test integrals and centroids
        for d in l:
            self.pd.set_weights(np.array([d["w"]]))

            ig = self.pd.integrals()
            self.assertAlmostEqual(ig[ 0 ], d["i"], places=5)

            cs = self.pd.centroids()
            for (v,e) in zip( cs[0], d["c"] ):
                self.assertAlmostEqual(v, e, places=5)

    def test_derivatives(self):
        self.pd.set_positions(np.array([[0.0, 0.0],[1.0, 0.0]]))
        self.pd.set_weights(np.array([1.0, 2.0]))
        check_ders( self, self.pd )

class TestArfd_2D_1p5(unittest.TestCase):
    def setUp(self):
        rf = RadialFuncArfd(
            lambda r: ( ( 1 - r * r ) * ( r < 1 ) ) ** 2.5, # value
            lambda w: 1 / w ** 0.5, # input scaling
            lambda w: w ** 2.5, # output scaling

            lambda r: 2.5 * ( ( 1 - r * r ) * ( r < 1 ) ) ** 1.5, # value for the der wrt weight
            lambda w: 1 / w ** 0.5, # input scaling for the der wrt weight
            lambda w: w ** 1.5, # output scaling for the der wrt weight

            [ 1 ], # stops (radii values where we may have discontinuities)
            1e-8 # precision
        )

        # set up a domain, with only 1 dirac
        domain = ConvexPolyhedraAssembly()
        domain.add_box([0.0, 0.0], [2.0, 1.0])

        self.pd = PowerDiagram(domain=domain, radial_func=rf)

    def test_integrals(self):
        self.pd.set_positions(np.array([[0.0, 0.0]]))

        # weights and expected values (w = weight, i = integral, c = centroid)
        l = [
            { "w":  10.0, "i":    417.0399, "c": [ 0.815858, 0.476057 ]     },
            { "w":  20.0, "i":   2902.3809, "c": [ 0.911708, 0.488773 ]     },
            { "w": 100.0, "i": 191826.6661, "c": [ 0.983124, 0.497884 ]     },
        ]

        # NumberForm[ N[ Integrate[ Integrate[ (  10 - ( x * x + y * y ) ) ^ 2.5, { x, 0, 2 } ], { y, 0, 1 } ], 10 ], 10 ] => 417.0399438275571
        # NumberForm[ N[ Integrate[ Integrate[ (  20 - ( x * x + y * y ) ) ^ 2.5, { x, 0, 2 } ], { y, 0, 1 } ], 10 ], 10 ] => 2902.3809
        # NumberForm[ N[ Integrate[ Integrate[ ( 100 - ( x * x + y * y ) ) ^ 2.5, { x, 0, 2 } ], { y, 0, 1 } ], 10 ], 10 ] => 191826.6661

        # NumberForm[ N[ Integrate[ Integrate[ x * (  10 - ( x * x + y * y ) ) ^ 2.5, { x, 0, 2 } ], { y, 0, 1 } ] / Integrate[ Integrate[ (  10 - ( x * x + y * y ) ) ^ 2.5, { x, 0, 2 } ], { y, 0, 1 } ], 10 ], 10 ]
        # NumberForm[ N[ Integrate[ Integrate[ y * (  10 - ( x * x + y * y ) ) ^ 2.5, { x, 0, 2 } ], { y, 0, 1 } ] / Integrate[ Integrate[ (  10 - ( x * x + y * y ) ) ^ 2.5, { x, 0, 2 } ], { y, 0, 1 } ], 10 ], 10 ]

        # NumberForm[ N[ Integrate[ Integrate[ x * (  20 - ( x * x + y * y ) ) ^ 2.5, { x, 0, 2 } ], { y, 0, 1 } ] / Integrate[ Integrate[ (  20 - ( x * x + y * y ) ) ^ 2.5, { x, 0, 2 } ], { y, 0, 1 } ], 10 ], 10 ]
        # NumberForm[ N[ Integrate[ Integrate[ y * (  20 - ( x * x + y * y ) ) ^ 2.5, { x, 0, 2 } ], { y, 0, 1 } ] / Integrate[ Integrate[ (  20 - ( x * x + y * y ) ) ^ 2.5, { x, 0, 2 } ], { y, 0, 1 } ], 10 ], 10 ]

        # NumberForm[ N[ Integrate[ Integrate[ x * ( 100 - ( x * x + y * y ) ) ^ 2.5, { x, 0, 2 } ], { y, 0, 1 } ] / Integrate[ Integrate[ ( 100 - ( x * x + y * y ) ) ^ 2.5, { x, 0, 2 } ], { y, 0, 1 } ], 10 ], 10 ]
        # NumberForm[ N[ Integrate[ Integrate[ y * ( 100 - ( x * x + y * y ) ) ^ 2.5, { x, 0, 2 } ], { y, 0, 1 } ] / Integrate[ Integrate[ ( 100 - ( x * x + y * y ) ) ^ 2.5, { x, 0, 2 } ], { y, 0, 1 } ], 10 ], 10 ]

        # test integrals and centroids
        for d in l:
            self.pd.set_weights(np.array([d["w"]]))

            ig = self.pd.integrals()
            self.assertAlmostEqual(ig[ 0 ], d["i"], places=2)

            cs = self.pd.centroids()
            for (v,e) in zip( cs[0], d["c"] ):
                self.assertAlmostEqual(v, e, delta=1e-3)

    def test_derivatives(self):
        self.pd.set_positions(np.array([[0.0, 0.0],[1.0, 0.0]]))
        self.pd.set_weights(np.array([1.0, 2.0]))
        check_ders( self, self.pd )


if __name__ == '__main__':
    unittest.main()
