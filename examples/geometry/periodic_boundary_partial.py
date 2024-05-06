from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import OptimalTransport, PowerDiagram
from scipy.sparse import csr_matrix
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
def get_pd( dw = 0.0, dx = 0.0, dy = 0.0 ):
    positions = np.array( [ [ 0.25 + dx, 0.25 + dy ], [ 0.75, 0.5 ] ] )
    weights = [ dw, 0.0 ]

    return PowerDiagram( positions, weights = weights )

def measure( dw = 0.0, dx = 0.0, dy = 0.0 ):
    return get_pd( dw, dx, dy ).integrals()[ 0 ]

def centrox( dw = 0.0, dx = 0.0, dy = 0.0 ):
    return get_pd( dw, dx, dy ).centroids()[ 0, 0 ]

def centroy( dw = 0.0, dx = 0.0, dy = 0.0 ):
    return get_pd( dw, dx, dy ).centroids()[ 0, 1 ]


def test_der():
    # nb_dims = 2
    pd = get_pd()
    pd.display_vtk( "pb.vtk" )

    print( "measure", measure() )
    print( "centrox", centrox() )
    print( "centroy", centroy() )

    mvs = pd.der_centroids_and_integrals_wrt_weight_and_positions()
    # print( mvs.m_values, mvs.m_columns, mvs.m_offsets )
    M = csr_matrix( ( mvs.m_values, mvs.m_columns, mvs.m_offsets ) ).todense()
    #M = np.resize( M, [ 2 * M.shape[ 0 ], M.shape[ 1 ] ] )

    N = np.zeros( [ 6, 3 ] )
    N[ 3:6, 0:3 ] = M[ 0:3, 0:3 ]

    eps = 1e-6
    N[ 0, 0 ] = ( centrox( dx = eps ) - centrox() ) / eps
    N[ 0, 1 ] = ( centrox( dy = eps ) - centrox() ) / eps
    N[ 0, 2 ] = ( centrox( dw = eps ) - centrox() ) / eps
    N[ 1, 0 ] = ( centroy( dx = eps ) - centroy() ) / eps
    N[ 1, 1 ] = ( centroy( dy = eps ) - centroy() ) / eps
    N[ 1, 2 ] = ( centroy( dw = eps ) - centroy() ) / eps
    N[ 2, 0 ] = ( measure( dx = eps ) - measure() ) / eps
    N[ 2, 1 ] = ( measure( dy = eps ) - measure() ) / eps
    N[ 2, 2 ] = ( measure( dw = eps ) - measure() ) / eps

    print( N )

test_der()
