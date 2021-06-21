from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import PowerDiagram
import matplotlib.pyplot as plt
import numpy as np

positions = np.array( [
    [ 0.0, 0.5 ],
    [ 1.0, 0.5 ],
] )

domain = ConvexPolyhedraAssembly()
domain.add_box([0, 0], [1, 1])

# diracs
pd = PowerDiagram( positions, domain = domain )
l = []
for i in range( 10000 ):
    l.append( pd.centroids( 0.5 ) )
l = np.vstack( l )

plt.plot( l[ :, 0 ], l[ :, 1 ], '.' )
plt.show()
