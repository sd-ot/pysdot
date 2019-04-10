from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from pysdot import OptimalTransport
import numpy as np

domain = ConvexPolyhedraAssembly()
domain.add_box( [ 0, 0 ], [ 1, 10 ] )

R = 0.5

ot = OptimalTransport(domain, RadialFuncInBall())
ot.set_masses( np.array( [ np.pi * R**2 / 2 ] ) )
ot.set_weights( np.array( [ R**2 ] ) )

cpt = 0
for y in np.linspace( 0, -20, 100 ):
    ot.set_positions( np.array( [ [ 0.5, y ] ] ) )
    ot.update_weights()

    r = ot.get_weights()[ 0 ] ** 0.5
    a = np.arcsin( R / r )

    e = 0.5 * r * ( np.pi / 2 * np.sin( a ) - a / np.sin( a ) + np.cos( a ) )
    f = 0.5 * ( np.cos( a ) + a / np.sin( a ) )
    yp = y + r * f

    print( y, yp )

    ot.display_vtk( "lc_{}.vtk".format( cpt ) )
    cpt += 1

