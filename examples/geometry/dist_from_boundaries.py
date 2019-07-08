from pysdot import PowerDiagram
import pylab as plt
import numpy as np

positions = np.random.rand( 10, 2 )

# diracs
pd = PowerDiagram( positions )

points = []
for y in np.linspace(0, 1, 100):
    for x in np.linspace(0, 1, 100):
        points.append( [ x, y ] )

dist = pd.distances_from_boundaries( np.array( points ) )
dist = dist.reshape( ( 100, 100 ) )
print( dist.shape )

plt.imshow( dist )
plt.colorbar()
plt.show()
