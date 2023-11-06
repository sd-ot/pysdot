from pysdot.domain_types import ConvexPolyhedraAssembly, ScaledImage
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
            pd = PowerDiagram(domain=self.domain)
            pd.set_positions(np.random.rand(nb_diracs, 2))
            pd.set_weights(np.ones(nb_diracs))

            # integrals
            areas = pd.integrals()
            self.assertAlmostEqual(np.sum(areas), 1.0)

    def test_unit(self):
        pd = PowerDiagram(domain=self.domain)
        pd.set_positions([[0.0, 0.0]])
        pd.set_weights([0.0])

        areas = pd.integrals()
        self.assertAlmostEqual(areas[0], 1.0)

        centroids = pd.centroids()
        self.assertAlmostEqual(centroids[0][0], 0.5)
        self.assertAlmostEqual(centroids[0][1], 0.5)

    def test_image_6(self):
        img = np.empty( [6, 1, 1] )
        img[ :, 0, 0 ] = np.array([ 2, 3, 4, 5, 6, 7 ])
        pd = PowerDiagram(
            domain=ScaledImage(np.array([0, 0]), np.array([1, 1]), img),
            positions=[
                [0.25, 0.25],
                [0.75, 0.25],
                [0.25, 0.75],
                [0.75, 0.75],
            ]
        )

        areas = pd.integrals()
        self.assertAlmostEqual(areas[0], 1.28125)
        self.assertAlmostEqual(areas[1], 2.46875)
        self.assertAlmostEqual(areas[2], 2.84375)
        self.assertAlmostEqual(areas[3], 4.40625)

        # Integrate[Integrate[x*(2+3*x+4*y+5*x*x+6*x*y+7*y*y),{x,0.0,0.5}],{y,0.0,0.5}]/Integrate[Integrate[2+3*x+4*y+5*x*x+6*x*y+7*y*y,{x,0.0,0.5}],{y,0.0,0.5}] -> 0.2784552845528455284552845528455284552845528455284552845528455284
        # Integrate[Integrate[x*(2+3*x+4*y+5*x*x+6*x*y+7*y*y),{x,0.5,1.0}],{y,0.0,0.5}]/Integrate[Integrate[2+3*x+4*y+5*x*x+6*x*y+7*y*y,{x,0.5,1.0}],{y,0.0,0.5}] -> 0.7753164556962025316455696202531645569620253164556962025316455696
        # Integrate[Integrate[x*(2+3*x+4*y+5*x*x+6*x*y+7*y*y),{x,0.0,0.5}],{y,0.5,1.0}]/Integrate[Integrate[2+3*x+4*y+5*x*x+6*x*y+7*y*y,{x,0.0,0.5}],{y,0.5,1.0}] -> 0.2683150183150183150183150183150183150183150183150183150183150183
        # Integrate[Integrate[x*(2+3*x+4*y+5*x*x+6*x*y+7*y*y),{x,0.5,1.0}],{y,0.5,1.0}]/Integrate[Integrate[2+3*x+4*y+5*x*x+6*x*y+7*y*y,{x,0.5,1.0}],{y,0.5,1.0}] -> 0.7677304964539007092198581560283687943262411347517730496453900709

        # Integrate[Integrate[y*(2+3*x+4*y+5*x*x+6*x*y+7*y*y),{x,0.0,0.5}],{y,0.0,0.5}]/Integrate[Integrate[2+3*x+4*y+5*x*x+6*x*y+7*y*y,{x,0.0,0.5}],{y,0.0,0.5}] -> 0.286585
        # Integrate[Integrate[y*(2+3*x+4*y+5*x*x+6*x*y+7*y*y),{x,0.5,1.0}],{y,0.0,0.5}]/Integrate[Integrate[2+3*x+4*y+5*x*x+6*x*y+7*y*y,{x,0.5,1.0}],{y,0.0,0.5}] -> 0.2753164556962025316455696202531645569620253164556962025316455696
        # Integrate[Integrate[y*(2+3*x+4*y+5*x*x+6*x*y+7*y*y),{x,0.0,0.5}],{y,0.5,1.0}]/Integrate[Integrate[2+3*x+4*y+5*x*x+6*x*y+7*y*y,{x,0.0,0.5}],{y,0.5,1.0}] -> 0.779304
        # Integrate[Integrate[y*(2+3*x+4*y+5*x*x+6*x*y+7*y*y),{x,0.5,1.0}],{y,0.5,1.0}]/Integrate[Integrate[2+3*x+4*y+5*x*x+6*x*y+7*y*y,{x,0.5,1.0}],{y,0.5,1.0}] -> 0.7724586288416075650118203309692671394799054373522458628841607565

        centroids = pd.centroids()
        self.assertAlmostEqual(centroids[0,0], 0.278455284552845528455284552845)
        self.assertAlmostEqual(centroids[1,0], 0.775316455696202531645569620253)
        self.assertAlmostEqual(centroids[2,0], 0.268315018315018315018315018315)
        self.assertAlmostEqual(centroids[3,0], 0.767730496453900709219858156028)

        self.assertAlmostEqual(centroids[0,1], 0.2865853658536585              )
        self.assertAlmostEqual(centroids[1,1], 0.275316455696202531645569620253)
        self.assertAlmostEqual(centroids[2,1], 0.779304                        )
        self.assertAlmostEqual(centroids[3,1], 0.772458628841607565011820330969)

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
            pd = PowerDiagram(domain=self.domain)
            pd.set_positions(np.random.rand(nb_diracs, 3))
            pd.set_weights(np.ones(nb_diracs))

            # integrals
            areas = pd.integrals()
            self.assertAlmostEqual(np.sum(areas), 1.0)

    def test_unit(self):
        pd = PowerDiagram(domain=self.domain)
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
