from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import OptimalTransport
import numpy as np

# # domain
domain = ConvexPolyhedraAssembly()
domain.add_box([0, 0], [1, 1])

# diracs
ot = OptimalTransport(domain)
ot.set_positions(np.random.rand(1000, 2))

# solve
ot.adjust_weights()

# display
ot.display_vtk( "results/pd.vtk" )
