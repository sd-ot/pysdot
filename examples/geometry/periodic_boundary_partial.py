from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import OptimalTransport
import numpy as np

nb_diracs = 10

positions = np.random.rand( nb_diracs, 3 )
masses = np.ones( nb_diracs ) / nb_diracs

domain = ConvexPolyhedraAssembly()
domain.add_box([-1, -1, 0], [2, 2, 1])

# we have to specify that dirac masses because by default, sdot takes ones * measure( domain ) / nb_diracs
# and that's not what we want in this case
ot = OptimalTransport( positions, domain = domain, masses = masses )

# first arg of add_replication is a translation
for x in [ -1, 0, 1 ]:
    for y in [ -1, 0, 1 ]:
        if x or y:
            ot.pd.add_replication( [ x, y, 0 ] )

ot.adjust_weights()

ot.display_vtk( "pb.vtk" )
