from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncEntropy
from pysdot.radial_funcs import RadialFuncInBall
from pysdot.radial_funcs import RadialFuncUnit
from pysdot import OptimalTransport
import numpy as np
import unittest


class TestOptimalTransport(unittest.TestCase):
    def setUp(self):
        self.domain = ConvexPolyhedraAssembly()
        self.domain.add_box([0, 0], [1, 1])

    def test_base_ot(self, nb_diracs=1000):
        for _ in range(100):
            ot = OptimalTransport(self.domain)

            # diracs
            ot.set_positions(np.random.rand(nb_diracs, 2))
            ot.set_weights(np.ones(nb_diracs))

            # optimal weights
            ot.adjust_weights()

            # integrals
            areas = ot.pd.integrals()
            self.assertAlmostEqual(np.min(areas), 1.0 / nb_diracs, places=6)
            self.assertAlmostEqual(np.max(areas), 1.0 / nb_diracs, places=6)

            # ot.pd.display_vtk("results/vtk/pd.vtk")

    def test_ball_cut(self, nb_diracs=100):
        for _ in range(10):
            ot = OptimalTransport(self.domain, radial_func=RadialFuncInBall())

            positions = np.random.rand(nb_diracs, 2)
            positions[:, 1] *= 0.5

            radius = 0.25 / nb_diracs**0.5
            mass = np.pi * radius**2

            # diracs
            ot.set_positions(positions)
            ot.set_weights(np.ones(nb_diracs) * radius**2)
            ot.set_masses(np.ones(nb_diracs) * mass)

            # optimal weights
            ot.adjust_weights()

            ot.pd.display_vtk("results/pd.vtk")

            # integrals
            areas = ot.pd.integrals()

            self.assertAlmostEqual(np.min(areas), mass, places=6)
            self.assertAlmostEqual(np.max(areas), mass, places=6)


if __name__ == '__main__':
    np.random.seed(1)
    unittest.main()
