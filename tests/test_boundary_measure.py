# from pysdot.domain_types import ConvexPolyhedraAssembly
# from pysdot.radial_funcs import RadialFuncInBall
# from pysdot import OptimalTransport
# import matplotlib.pyplot as plt
# import numpy as np

# def test_discr( ot, n, coeff_r, cpt ):
#     da = np.pi / 2 / ( n - 1 )
#     area_per_small_disc = np.pi / 2 * ( np.sin( da ) / 2 )**2
#     area_large_disc = np.pi / 4 - n * area_per_small_disc
#     r = coeff_r * ( 4 * area_large_disc / np.pi )**0.5

#     masses = [ area_large_disc ]
#     positions = [ [ 0, 0 ] ]
#     for i in range( n ):
#         a = da * i
#         masses.append( area_per_small_disc )
#         positions.append( [ r * np.cos( a ), r * np.sin( a ) ] )

#     ot.set_positions( np.array( positions ) )
#     if cpt == 0:
#         ot.set_weights( np.array( masses ) / np.pi * 0.25 )
#     ot.set_masses( np.array( masses ) )
#     ot.max_iter = 100

#     ot.adjust_weights( relax=0.5 )

#     ot.display_vtk( "lc_{}.vtk".format( cpt ) )

#     return ot.pd.der_boundary_integral()


# domain = ConvexPolyhedraAssembly()
# domain.add_box( [ 0, 0 ], [ 10, 10 ] )

# crb = {}
# for n in range( 30, 31 ):
#     ot = OptimalTransport( domain, RadialFuncInBall() )
#     k = "{}".format( n )
#     crb[ k ] = []
#     cpt = 0
#     for coeff_r in [ 0.9875 ]: # np.linspace( 0.9875, 0.4, 120 ):
#         e = test_discr( ot, n, coeff_r, cpt )
#         print( e )
#         # crb[ k ].append( [ coeff_r, e ] )
#         # cpt += 1

# # for key, val in crb.items():
# #     val = np.array( val )
# #     plt.plot( val[ :, 0 ], val[ :, 1 ] )
# # plt.show()
