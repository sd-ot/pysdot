from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncEntropy
from pysdot import PowerDiagram
import numpy as np
import unittest


class TestIntegrate_2D(unittest.TestCase):
    def setUp(self):
        self.domain = ConvexPolyhedraAssembly()
        self.domain.add_box([0, 0], [1, 1])

    def test_sum_area(self, nb_diracs=100):
        for _ in range(10):
            # diracs
            pd = PowerDiagram(self.domain)
            pd.set_positions(np.random.rand(nb_diracs, 2))
            pd.set_weights(np.ones(nb_diracs))

            # integrals
            areas = pd.integrals()
            self.assertAlmostEqual(np.sum(areas), 1.0)

    def test_unit(self):
        pd = PowerDiagram(self.domain)
        pd.set_positions([[0.0, 0.0]])
        pd.set_weights([0.0])

        areas = pd.integrals()
        self.assertAlmostEqual(areas[0], 1.0)

        centroids = pd.centroids()
        self.assertAlmostEqual(centroids[0][0], 0.5)
        self.assertAlmostEqual(centroids[0][1], 0.5)

    def test_gaussian(self):
        # wolfram: N[ Integrate[ Integrate[ Exp[ ( 0 - x*x - y*y ) / 1 ], { x, -0.5, 0.5 } ], { y, -0.5, 0.5 } ] ]
        # wolfram: N[ Integrate[ Integrate[ x * Exp[ ( 0 - x*x - y*y ) / 0.1 ], { x, 0, 1 } ], { y, 0, 1 } ] ] / N[ Integrate[ Integrate[ Exp[ ( 0 - x*x - y*y ) / 0.1 ], { x, 0, 1 } ], { y, 0, 1 } ] ]
        def tg(position, w, eps, ei, ec):
            pd = PowerDiagram(
                radial_func=RadialFuncEntropy(eps),
                domain=self.domain
            )
            pd.set_positions([position])
            pd.set_weights([w])

            res = pd.integrals()
            self.assertAlmostEqual(res[0], ei, 5)

            res = pd.centroids()
            self.assertAlmostEqual(res[0][0], ec[0], 5)
            self.assertAlmostEqual(res[0][1], ec[1], 5)

        tg([0.5, 0.5], w=0.0, eps=1.0, ei=0.851121, ec=[0.500000, 0.500000])
        tg([0.5, 0.5], w=1.0, eps=1.0, ei=2.313590, ec=[0.500000, 0.500000])
        tg([0.5, 0.5], w=0.0, eps=2.0, ei=0.921313, ec=[0.500000, 0.500000])
        tg([0.5, 0.5], w=1.0, eps=2.0, ei=1.518990, ec=[0.500000, 0.500000])

        tg([0.0, 0.0], w=0.0, eps=1.0, ei=0.557746, ec=[0.423206, 0.423206])
        tg([0.0, 0.0], w=1.0, eps=1.0, ei=1.516110, ec=[0.423206, 0.423206])
        tg([0.0, 0.0], w=0.0, eps=2.0, ei=0.732093, ec=[0.459862, 0.459862])
        tg([0.0, 0.0], w=1.0, eps=2.0, ei=1.207020, ec=[0.459862, 0.459862])

        tg([0.0, 0.0], w=0.0, eps=0.1, ei=.0785386, ec=[0.178406, 0.178406])


class TestIntegrate_3D(unittest.TestCase):
    def setUp(self):
        self.domain = ConvexPolyhedraAssembly()
        self.domain.add_box([0, 0, 0], [1, 1, 1])

    def test_sum_area(self, nb_diracs=100):
        for _ in range(10):
            # diracs
            pd = PowerDiagram(self.domain)
            pd.set_positions(np.random.rand(nb_diracs, 3))
            pd.set_weights(np.ones(nb_diracs))

            # integrals
            areas = pd.integrals()
            self.assertAlmostEqual(np.sum(areas), 1.0)

    def test_unit(self):
        pd = PowerDiagram(self.domain)
        pd.set_positions([[0.0, 0.0, 0.0]])
        pd.set_weights([0.0])

        areas = pd.integrals()
        self.assertAlmostEqual(areas[0], 1.0)

        centroids = pd.centroids()
        self.assertAlmostEqual(centroids[0][0], 0.5)
        self.assertAlmostEqual(centroids[0][1], 0.5)
        self.assertAlmostEqual(centroids[0][2], 0.5)

if __name__ == '__main__':
    unittest.main()
