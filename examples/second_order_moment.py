from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import PowerDiagram
import numpy as np

positions = np.array( [
    [ 0.25, 0.5 ],
    [ 0.75, 0.5 ]
] )

# domain = ConvexPolyhedraAssembly()
# domain.add_box( [ 0, 0 ], [ 0.5, 1 ], 1 )
# domain.add_box( [ 0.5, 0 ], [ 1, 1 ], 0.1 )

# diracs
pd = PowerDiagram( positions )
print( np.sum( pd.second_order_moments() ) )

positions[ 0, 0 ] = 0.0

pd = PowerDiagram( positions )
print( np.sum( pd.second_order_moments() ) )
