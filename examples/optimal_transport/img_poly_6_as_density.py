from pysdot.domain_types import ScaledImage
from pysdot import OptimalTransport
# from matplotlib import pyplot
import numpy as np

# initial positions
n = 20
positions = []
for y in range( n ):
    for x in range( n ):
        positions.append( [ ( x + 0.5 ) / n, ( y + 0.5 ) / n ] )

img = np.zeros( [ 6, 4, 1 ] )
img[ 0, 0, 0 ] = 1.1
img[ 0, 1, 0 ] = 2.6
img[ 0, 2, 0 ] = 1.6
img[ 0, 3, 0 ] = 0.6
# img[ 5, 0, 0 ] = + 5
# img[ 5, 1, 0 ] = - 1

domain = ScaledImage( [ 0, 0 ], [ 1, 1 ], img )
print( "domain.measure:", domain.measure() )

# x = []
# for y in np.linspace( 0, 1, 2000 ):
#     x.append( domain.coeff_at( [ 0.5, y ] ) )

# print( np.mean( x ) )

# pyplot.plot( x )
# pyplot.show()

ot = OptimalTransport( positions = positions, domain = domain, verbosity = 2 )
ot.adjust_weights()

ot.pd.display_vtk( "results/pd.vtk", centroids=True )
