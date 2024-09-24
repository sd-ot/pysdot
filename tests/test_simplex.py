import numpy as np

from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import OptimalTransport
from pysdot import PowerDiagram

def solve_OT_problem(x,v,om):
    n = np.shape(v) # number of seeds

    # Error tolerance for the OT solver
    err_tol = 1e-3*np.min(v)

    # Generate initial weights such that all cells have positive area
    w0 = np.zeros(n)

    # Initialise the optimal transport problem
    ot = OptimalTransport(positions = x, masses = v, weights = w0, domain = om,
                          obj_max_dm = err_tol, verbosity = 2,
                          linear_solver = 'Scipy')

    # Solve the optimal transport problem
    ot.adjust_weights()

    # Get cell data
    volumes = ot.pd.integrals() # volumes of the cells
    w = ot.get_weights() # weights

    # Difference between the true volumes and target volumes
    vol_error = np.max( np.abs(v-volumes) )

    # Error message if dv > tolerance
    if vol_error > err_tol:
        print('Maximum error in volumes ',vol_error)
        raise('Volume tolerance not met')

    return w, volumes

nodes = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.],[0.,0.,1.],[1.,0.,1.],[0.,1.,1.],[1.,1.,1.]])
tetrahedra = np.array([[0,7,2,6],[0,5,7,4],[4,6,7,0],[0,3,7,1],[0,3,2,7],[0,1,7,5]])

om = ConvexPolyhedraAssembly()
for tetrahedron in tetrahedra:
    om.add_simplex( nodes[ tetrahedron ] )

print( om.measure() ) # This should be 1

np.random.seed( 20 )


# Numer of seeds
n = 40
x = np.random.rand(n,3)
v = np.ones( n ) * om.measure() / n

# print( x ) # This should be 1

# Solve the optimal transport problem
w, volumes = solve_OT_problem( x, v, om )

# Print the volume error
print(np.abs(volumes-v))
