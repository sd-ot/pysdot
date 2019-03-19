import py_power_diagram_test_context
import py_power_diagram as pd
import numpy as np
np.random.seed( 1 )

nb_diracs = 1000

domain = pd.domain_types.ConvexPolyhedraAssembly()
domain.add_box( [ 0, 0 ], [ 1, 1 ] )

# diracs
positions = np.random.rand( nb_diracs, 2 )
weights = np.ones( nb_diracs )

# optimal weights
new_weights = pd.optimal_transport_2( "1", positions, weights, domain )

# integrals
# areas = pd.get_integrals( "1", positions, new_weights, domain )
# assertAlmostEqual( np.min( areas ), 1.0 / nb_diracs, places = 6 )
# assertAlmostEqual( np.max( areas ), 1.0 / nb_diracs, places = 6 )

pd.display_vtk( "vtk/pd.vtk", positions, new_weights, domain )
