from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import OptimalTransport
import numpy as np

positions = np.random.rand(200,2)

# diracs
ot = OptimalTransport()
ot.set_positions(np.array(positions))
ot.set_weights(np.ones(ot.get_positions().shape[0]))
ot.verbosity = 1

# solve
ot.adjust_weights()

# display
ot.display_vtk( "results/pd.vtk" )

# print( ot.pd.display_html() )

