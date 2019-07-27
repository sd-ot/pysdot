from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.domain_types import ScaledImage
from pysdot.radial_funcs import RadialFuncInBall
from pysdot.util import FastMarching
from pysdot import OptimalTransport
from pysdot import PowerDiagram

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import numpy as np

# if der = True, also returns a sparse matrix representing the Jacobian of the areas with respect to psi
def laguerre_areas(domain, Y, psi, der=False):
    pd = PowerDiagram(Y, -psi, domain)
    if der:
        N = len(psi)
        mvs = pd.der_integrals_wrt_weights()
        return mvs.v_values, csr_matrix((-mvs.m_values, mvs.m_columns, mvs.m_offsets), shape=(N, N))
    else:
        return pd.integrals()

def make_square(box=[0, 0, 1, 1]):
    domain = ConvexPolyhedraAssembly()
    domain.add_box([box[0], box[1]], [box[2], box[3]])
    return domain

def optimal_transport(domain, Y, nu, psi0=None, verbose=False, maxerr=1e-6, maxiter=20):
    if psi0 is None:
        psi0 = np.zeros(len(nu))
        
    def F(psip):
        g,h = laguerre_areas(domain, Y, np.hstack((psip,0)), der=True)
        return g[0:-1], h[0:-1,0:-1]
    
    psip = psi0[0:-1] - psi0[-1]
    nup = nu[0:-1]
    g,h = F(psip)
    for it in range(maxiter):
        err = np.linalg.norm(nup - g)
        if verbose:
            print("it %d: |err| = %g" % (it, err))
        if err <= maxerr:
            break
        d = spsolve(h, nup - g)
        t = 1.
        psip0 = psip.copy()
        while True:
            psip = psip0 + t * d
            g,h = F(psip)
            if np.min(g) > 0:
                break
            else:
                t = t/2
    return np.hstack((psip,0))

# computes the centroid of the Laguerre cells intersected with the domain, and returns it as a Nx2 array
def laguerre_centroids(domain, Y, psi):
    return PowerDiagram(Y, -psi, domain).centroids()

def optimal_quantization(domain, Y, tau=.1, niter=50, nstep_to_disp=0):
    d = np.arange(nstep_to_disp) * (niter - 1) // (nstep_to_disp - 1 + (nstep_to_disp==1))
    for i in range(niter):
        m = domain.measure() / Y.shape[0]
        psi = optimal_transport(domain, Y, m * np.ones(Y.shape[0]), verbose=False)
        # if i in d:
        #     clear_output(True)
        #     laguerre_draw(domain, Y, psi, disp_centroids=True)
        if i + 1 == niter:
          break
        B = laguerre_centroids(domain, Y, psi)
        Y = Y + tau * (B - Y)
    return Y, psi

def partial_optimal_transport(domain, Y, nu, psi0 = None):
    if psi0 == None:
        psi = -np.sqrt(nu)
    else:
        psi = -psi0.copy()
    ot = OptimalTransport(Y, -psi, domain, radial_func=RadialFuncInBall())
    ot.set_masses(nu)
    ot.adjust_weights()
    return ot.pd

# constants
N = 100 # nb diracs
b = 0.33 # initial box size
m = b**2 / N
nu = m * np.ones( N ) # masses

# initial positions
Y, psi = optimal_quantization(make_square([0,0,b,b]), b * np.random.rand(N, 2))

# domain
img = np.array([
    [1, 0, 1],
    [1, 1, 1],
    [1, 0, 1]
])
domain = ScaledImage([0, 0], [1, 1], img)

# fast marching
fm = FastMarching(domain, [[0.9,0.1]], .01)

# iterations
Y_hist = []
psi_hist = []
for num_time_step in range(50):
    for n in range(Y.shape[0]):
        d = 0.9 * m**0.5
        Y[n, :] += d * fm.grad(Y[n, :], 2 * m**0.5)
    pd = partial_optimal_transport(domain, Y, nu)

    # if num_time_step % 5 == 0:
    #     Y_hist.append(Y)
    #     psi_hist.append(pd.get_weights())
    pd.display_vtk("results/pd_{}.vtk".format(num_time_step))

    Y = pd.centroids()


# pd = PowerDiagram(Y_hist, psi_hist, domain, RadialFuncInBall())
# display(pd.display_jupyter(disp_ids=False))