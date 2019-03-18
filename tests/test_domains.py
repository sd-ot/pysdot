from pysdot.domain_types import ConvexPolyhedraAssembly
import unittest


class TestConvexPolyhedraAssembly(unittest.TestCase):
    def test_measure(self):
        domain = ConvexPolyhedraAssembly()
        domain.add_box([0, 0], [2, 1])
        self.assertAlmostEqual(domain.measure(), 2.0)

if __name__ == '__main__':
    unittest.main()
