from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncEntropy
from pysdot.radial_funcs import RadialFuncUnit
from pysdot import PowerDiagram
import numpy as np
import unittest


class TestDerIntegrate_2D_weights(unittest.TestCase):
    def setUp(self):
        self.domain = ConvexPolyhedraAssembly()
        self.domain.add_box([0, 0], [2, 1])

    def test_unit(self):
        self._test_der([[0.5, 0.5], [1.5, 0.5]], [0, 0], RadialFuncUnit())

    def test_gaussian(self):
        # wolfram: x=-0.5; y=(1-u)*(0.5)-0.5; N[ Integrate[ Exp[ ( 0 - x*x - y*y ) / 1 ], { u, 0, 1 } ]] -> 0.718492
        self._test_der([[0.4, 0.5], [1.5, 0.5]], [1, 0], RadialFuncEntropy(1))
        self._test_der([[0.4, 0.5], [1.5, 0.5]], [1, 2], RadialFuncEntropy(1))

        N = 10
        for _ in range(50):
            positions = [
                [np.random.rand(), np.random.rand()] for i in range(N)
            ]
            weights = [np.random.rand() * 0.1 for i in range(N)]
            self._test_der(positions, weights, RadialFuncEntropy(2))

    def _test_der(self, positions, weights, rf):
        pd = PowerDiagram(self.domain, rf)
        pd.set_positions(np.array(positions, dtype=np.double))
        pd.set_weights(np.array(weights, dtype=np.double))

        num = pd.der_integrals_wrt_weights()
        if num.error:
            return

        ndi = len(positions)
        mat = np.zeros((ndi, ndi))
        for i in range(ndi):
            for o in range(num.m_offsets[i + 0], num.m_offsets[i + 1]):
                mat[i, num.m_columns[o]] = num.m_values[o]

        eps = 1e-6
        res = pd.integrals()
        delta = np.max(np.abs(mat)) * 200 * eps
        for i in range(ndi):
            pd.set_weights(np.array(
                [weights[j] + eps * (i == j) for j in range(ndi)]
            ))
            des = pd.integrals()
            der = (des - res) / eps
            for j in range(ndi):
                self.assertAlmostEqual(mat[i, j], der[j], delta=delta)


# class TestDerIntegrate_2D_weight_and_positions(unittest.TestCase):
#     def setUp(self):
#         self.domain = ConvexPolyhedraAssembly()
#         self.domain.add_box([0, 0], [2, 1])

#     def test_unit(self):
#         self._test_der([[0.5, 0.5], [1.5, 0.5]], [0, 0], RadialFuncUnit())
#         self._test_der([[0.5, 0.5], [1.5, 0.5]], [0, 0], RadialFuncUnit())

#     def _test_der(self, positions, weights, rf):
#         pd = PowerDiagram(self.domain, rf)
#         pd.set_positions(np.array(positions, dtype=np.double))
#         pd.set_weights(np.array(weights, dtype=np.double))

#         pd.display_vtk("der_int.vtk")

#         num = pd.der_integrals_wrt_weight_and_positions()
#         print("------------", num.error)
#         if num.error:
#             return

#         ndi = len(positions)
#         mat = np.zeros((ndi, ndi))
#         for i in range(ndi):
#             for o in range(num.m_offsets[i + 0], num.m_offsets[i + 1]):
#                 mat[i, num.m_columns[o]] = num.m_values[o]

#         print("------------",mat)

#         # eps = 1e-6
#         # res = pd.integrals()
#         # delta = np.max(np.abs(mat)) * 100 * eps
#         # for i in range(ndi):
#         #     pd.set_weights(np.array(
#         #         [weights[j] + eps * (i == j) for j in range(ndi)]
#         #     ))
#         #     des = pd.integrals()
#         #     der = (des - res) / eps
#         #     for j in range(ndi):
#         #         self.assertAlmostEqual(mat[i, j], der[j], delta=delta)


if __name__ == '__main__':
    unittest.main()
