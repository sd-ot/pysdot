from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import PowerDiagram
import numpy as np
import unittest


class TestIntegrate(unittest.TestCase):
    def setUp(self):
        self.domain = ConvexPolyhedraAssembly()
        self.domain.add_box([0, 0, 0], [1, 1, 1])

    def test_sum_area(self, nb_diracs=100):
        for _ in range(1):
            # diracs
            pd = PowerDiagram()
            pd.set_positions(np.random.rand(nb_diracs, 3))
            pd.set_weights(np.ones(nb_diracs))
            pd.set_domain(self.domain)

            # integrals
            areas = pd.integrals()
            self.assertAlmostEqual(np.sum(areas), 1.0)

# import py_power_diagram_test_context
# import numpy as np

# class TestIntegrate( unittest.TestCase ):
#     def setUp( self, nb_diracs = 100 ):
#         self.domain = pd.domain_types.ConvexPolyhedraAssembly()
#         self.domain.add_box( [ 0, 0 ], [ 1, 1 ] )

#     def test_sum_area( self, nb_diracs = 100 ):
#         for _ in range( 10 ):
#             # diracs
#             positions = np.random.rand( nb_diracs, 2 )
#             weights = np.ones( nb_diracs )

#             # integrals
#             areas = pd.get_integrals( "1", positions, weights, self.domain )
#             self.assertAlmostEqual( np.sum( areas ), 1.0 )

#     def test_unit( self ):
#         res = pd.get_integrals( "1", np.array( [[ 0.0, 0.0 ]] ), np.zeros( 1 ), self.domain )
#         self.assertAlmostEqual( res[ 0 ], 1.0 )

#         res = pd.get_centroids( "1", np.array( [[ 0.0, 0.0 ]] ), np.zeros( 1 ), self.domain )
#         self.assertAlmostEqual( res[ 0 ][ 0 ], 0.5 )
#         self.assertAlmostEqual( res[ 0 ][ 1 ], 0.5 )

#     def test_gaussian( self ):
#         # wolfram: N[ Integrate[ Integrate[ Exp[ ( 0 - x*x - y*y ) / 1 ], { x, -0.5, 0.5 } ], { y, -0.5, 0.5 } ] ]
#         # wolfram: N[ Integrate[ Integrate[ x * Exp[ ( 0 - x*x - y*y ) / 0.1 ], { x, 0, 1 } ], { y, 0, 1 } ] ] / N[ Integrate[ Integrate[ Exp[ ( 0 - x*x - y*y ) / 0.1 ], { x, 0, 1 } ], { y, 0, 1 } ] ]

#         self._test_gaussian_for( [ 0.5, 0.5 ], w=0, eps=1.0, exp_int=0.851121 , exp_ctr=[ 0.5     , 0.5      ] )
#         self._test_gaussian_for( [ 0.5, 0.5 ], w=1, eps=1.0, exp_int=2.31359  , exp_ctr=[ 0.5     , 0.5      ] )
#         self._test_gaussian_for( [ 0.5, 0.5 ], w=0, eps=2.0, exp_int=0.921313 , exp_ctr=[ 0.5     , 0.5      ] )
#         self._test_gaussian_for( [ 0.5, 0.5 ], w=1, eps=2.0, exp_int=1.51899  , exp_ctr=[ 0.5     , 0.5      ] )

#         self._test_gaussian_for( [ 0.0, 0.0 ], w=0, eps=1.0, exp_int=0.557746 , exp_ctr=[ 0.423206, 0.423206 ] )
#         self._test_gaussian_for( [ 0.0, 0.0 ], w=1, eps=1.0, exp_int=1.51611  , exp_ctr=[ 0.423206, 0.423206 ] )
#         self._test_gaussian_for( [ 0.0, 0.0 ], w=0, eps=2.0, exp_int=0.732093 , exp_ctr=[ 0.459862, 0.459862 ] )
#         self._test_gaussian_for( [ 0.0, 0.0 ], w=1, eps=2.0, exp_int=1.20702  , exp_ctr=[ 0.459862, 0.459862 ] )

#         self._test_gaussian_for( [ 0.0, 0.0 ], w=0, eps=0.1, exp_int=0.0785386, exp_ctr=[ 0.178406, 0.178406 ] )

#     def _test_gaussian_for( self, positions, w, eps, exp_int, exp_ctr ):
#         res = pd.get_integrals( "exp((w-r**2)/{})".format( eps ), np.array( [ positions ] ), np.zeros( 1 ) + w, self.domain )
#         self.assertAlmostEqual( res[ 0 ], exp_int, 5 )

#         res = pd.get_centroids( "exp((w-r**2)/{})".format( eps ), np.array( [ positions ] ), np.zeros( 1 ) + w, self.domain )
#         self.assertAlmostEqual( res[ 0 ][ 0 ], exp_ctr[ 0 ], 5 )
#         self.assertAlmostEqual( res[ 0 ][ 1 ], exp_ctr[ 1 ], 5 )

# der measures
# ( has_a_void_cell, m_offsets, m_columns, m_values, v_values ) = pd.get_der_integrations_wrt_weights( positions, weights, domain )
# # display
# pd.display_vtk( "vtk/pd.vtk", positions, weights, domain )
# domain.display_boundaries_vtk( "vtk/domain.vtk" )

if __name__ == '__main__':
    unittest.main()
