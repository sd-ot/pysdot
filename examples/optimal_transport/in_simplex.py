from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import OptimalTransport
import numpy as np

positions = np.random.rand(200,2)

domain = ConvexPolyhedraAssembly()
domain.add_convex_polyhedron([ [
    0.5 + 4 * np.cos( a ), # point X
    0.5 + 4 * np.sin( a ), # point Y
    np.cos( a + 2 * np.pi / 6 ), # normal X
    np.sin( a + 2 * np.pi / 6 )  # normal Y
] for a in np.linspace( 0, 2 * np.pi, 3, endpoint=False ) ])

# diracs
ot = OptimalTransport(domain=domain)
ot.set_positions(np.array(positions))
ot.verbosity = 2

# solve
ot.adjust_weights()

# display
ot.display_vtk( "results/pd.vtk" )
