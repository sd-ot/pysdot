import sys
sys.path.append( "/Users/hugo.leclerc/Code/pysdot" )
sys.path.append( "build/lib.macosx-11.0-arm64-cpython-313" )

from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncEntropy
from pysdot.domain_types import ScaledImage
from pysdot import OptimalTransport
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2214)

# positions
positions = [[-1,0.],[+1.,0.]]
weights = [0.,0.]

# domain
l = 2
t = np.linspace(-l,l,200)
x, y = np.meshgrid(t, t)

destination = np.exp( -( x**2 + y**2 ) / 2 ) / ( 2 * np.pi )
domain = ScaledImage([-l,-l], [+l,+l], destination)

ot = OptimalTransport( positions, weights = weights, domain = domain, radial_func = RadialFuncEntropy( eps = 2 ) )
print( ot.pd.integrals() )

# domain = ScaledImage([-l,-l], [+l,+l], destination)
# r = []
# for c in np.linspace( -l, l, 20 ):
#     domain = ConvexPolyhedraAssembly()
#     domain.add_box( [-l,-l], [c,c] )
#     # domain.add_box( [-1,-l], [+l,+l] )

#     # self.s = "exp((w-r**2)/{})".format(eps)
#     ot = OptimalTransport( positions, weights = weights, domain = domain, radial_func = RadialFuncEntropy( eps = 1 ) ) # 
#     # print( ot.pd.integrals() )
#     r.append( ot.pd.integrals()[ 0 ] )
# print( r )
# plt.plot( r )
# plt.show()
# ot.linear_solver = "Scipy"
# ot.verbosity = 2

# ot.adjust_weights()

# moving_points = [ot.get_positions()]

# energies = [energy(ot.get_positions())]
# for iter in range(max_iter):
#     if ot.verbosity >= 2:
#         print("iteration", iter)
#     density = RadialFuncEntropy(eps = -2*tau)
#     ot.set_radial_func(density)
#     ot.adjust_weights()
#     B = ot.get_centroids()
#     ot.set_positions(B)
#     moving_points.append(ot.get_positions())
#     energies.append(energy(ot.get_positions()))
# return (moving_points, energies)