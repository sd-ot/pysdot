from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import PowerDiagram
import numpy as np

positions = np.random.rand( 30, 2 )

domain = ConvexPolyhedraAssembly()
domain.add_box([-1, -1], [2, 2])

# diracs
pd = PowerDiagram( positions, domain = domain )
pd.add_replication( [ +1, 0 ] )
pd.add_replication( [ -1, 0 ] )
pd.display_vtk( "pb.vtk" )
