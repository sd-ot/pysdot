import sys, os
for d in os.listdir( "build" ):
    sys.path.append( os.path.join( "build", d ) )
sys.path.append( "." )

from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import CompressibleFunc
from pysdot import OptimalTransport
import numpy as np


# domain
domain = ConvexPolyhedraAssembly()
domain.add_box( [0, 0], [1, 1] )

positions = np.array([[0.0, 0.5], [1.0, 0.5]])

ot = OptimalTransport( positions = positions, domain = domain, radial_func = CompressibleFunc( kappa=1, gamma=0.5, g=9.81, f_cor = 1 ))
print( ot.pd.integrals() )
