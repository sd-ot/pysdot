from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import OptimalTransport
import numpy as np

# # domain
domain = ConvexPolyhedraAssembly()
domain.add_box([0, 0], [1, 1])

# diracs
ot = OptimalTransport(domain)
ot.set_positions(np.random.rand(10, 2))
ot.set_weights(np.ones(ot.get_positions().shape[0]))

print( ot.pd.display_html() )

# # solve
# ot.adjust_weights()

# # display
# ot.display_vtk( "results/pd.vtk" )

# print( ot.pd.display_html() )

