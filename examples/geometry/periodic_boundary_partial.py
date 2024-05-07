from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import OptimalTransport, PowerDiagram
from scipy.sparse import csr_matrix
from tabulate import tabulate
import numpy as np

# nb_diracs = 10

# positions = np.random.rand( nb_diracs, 3 )
# masses = np.ones( nb_diracs ) / nb_diracs

# domain = ConvexPolyhedraAssembly()
# domain.add_box([-1, -1, 0], [2, 2, 1])

# # we have to specify that dirac masses because by default, sdot takes ones * measure( domain ) / nb_diracs
# # and that's not what we want in this case
# ot = OptimalTransport( positions, domain = domain, masses = masses )

# # first arg of add_replication is a translation
# for x in [ -1, 0, 1 ]:
#     for y in [ -1, 0, 1 ]:
#         if x or y:
#             ot.pd.add_replication( [ x, y, 0 ] )

# ot.adjust_weights()

# ot.display_vtk( "pb.vtk" )

# print( ot.pd.der_centroids_and_integrals_wrt_weight_and_positions() )

def get_pd( dim, mod = "", eps = 0.0 ):
    positions = np.array( [ 
        [ 0.25 + ( mod == "x0" ) * eps, 0.25 + ( mod == "y0" ) * eps ] + [ 0.50 + ( mod == "z0" ) * eps ] * ( dim == 3 ), 
        [ 0.75 + ( mod == "x1" ) * eps, 0.50 + ( mod == "y1" ) * eps ] + [ 0.50 + ( mod == "z1" ) * eps ] * ( dim == 3 )
    ] )
    weights = [ ( mod == "w0" ) * eps, ( mod == "w1" ) * eps ]

    return PowerDiagram( positions, weights = weights )

def value( dim, index, cell, mod = "", eps = 0.0 ):
    pd = get_pd( dim, mod, eps )
    if index < dim:
        return pd.centroids()[ cell, index ]
    return pd.integrals()[ cell ]

def test_der():
    dim = 3
    pd = get_pd( dim )
    pd.display_vtk( "pb.vtk" )

    mvs = pd.der_centroids_and_integrals_wrt_weight_and_positions()
    M = csr_matrix( ( mvs.m_values, mvs.m_columns, mvs.m_offsets ) ).todense()

    eps = 1e-6
    values = []
    l1 = []
    l2 = []
    for i in range( 2 * ( dim + 1 ) ):
        c1 = []
        c2 = []
        for j in range( 2 * ( dim + 1 ) ):
            l = "wxyz"[ ( j + 1 ) % ( dim + 1 ) ]

            der = ( value( dim, i % ( dim + 1 ), i // ( dim + 1 ), f"{ l }{ j // ( dim + 1 ) }", eps ) - 
                    value( dim, i % ( dim + 1 ), i // ( dim + 1 ) ) 
            ) / eps

            values.append( [ M[ i, j ], der ] )
            c1.append( M[ i, j ] )
            c2.append( der )
        l1.append( c1 )
        l2.append( c2 )
    
    print( tabulate( values, floatfmt=".4f" ) )
    print( tabulate( l1, floatfmt=".4f" ) )
    print( tabulate( l2, floatfmt=".4f" ) )

test_der()

