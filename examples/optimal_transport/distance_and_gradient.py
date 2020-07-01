from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.domain_types import ScaledImage
from pysdot import PowerDiagram
import numpy as np
import matplotlib.pyplot as plt

from pysdot.radial_funcs import RadialFuncUnit
from pysdot import OptimalTransport

# Main function that computes optimal transport and associated quantities of interest.
# You can use semi-discrete optimal transport to measure the distance from a collection 
# weighted Dirac masses (discrete measure) to a density supported on some domain (continuous measure).
#
# density    = continuous measure, see make_square and make_image to create basic densities
# (Y,masses) = discrete measures, positions and weights
#               Y is of shape (N,d) and masses of shape (N,)
#               where d is the number of dimensions of the problem
#                     N is the number of diracs
def optimal_transport(density, Y, masses, psi0 = None, err=1e-8):
    center = (density.min_position() + density.max_position())/2
    halfsides = (density.max_position() - density.min_position())/2
    ratio = 1/np.max(np.abs((Y-center)/halfsides))
    psi = (1-ratio)*np.sum((Y-center)**2, axis=-1)
    ot = OptimalTransport(Y, psi, density, radial_func=RadialFuncUnit(), obj_max_dw=err)
    ot.set_masses(masses)
    ot.adjust_weights()
    return ot.pd

# Compute Wasserstein distance and associated gradient with respect to Y (discrete measure positions)
def wasserstein_distance_and_gradient(density, Y, masses):
    pd = optimal_transport(density, Y, masses)
    transport_cost = 0.5*np.sum(pd.second_order_moments())
    barycenters = pd.centroids()
    gradient = masses[:,None]*(Y-barycenters)
    return (transport_cost, gradient, pd)

# Flattened version of wasserstein distance in a given dimension 
def flat_wass(dens, y, masses, dim=2):
    assert y.ndim == 1, 'y should be one-dimensional'
    assert masses.ndim == 1, 'masses should be one-dimensional'
    assert masses.size == y.size // dim, 'masses size does not match with y size'
    N = y.size//dim
    (tc, grad, pd) = wasserstein_distance_and_gradient(dens, y.reshape(N,dim), masses)
    return tc, grad.flatten()

# Constructs a uniform density supported on a rectangle domain
# The box parameter should contain four floats: xmin, ymin, xmax, ymax
def make_square(box=[0, 0, 1, 1]):
    density = ConvexPolyhedraAssembly()
    density.add_box([box[0], box[1]], [box[2], box[3]])
    return density

# Constructs a density supported on a rectangle domain from discrete values sampled on a Cartesian grid.
# This may be used to generate a density from grayscale image data (values have to be positive).
def make_image(img, box=[0, 0, 1, 1]):
    img = img / ((box[2] - box[0]) * (box[3] - box[1]) * np.mean(img))
    return ScaledImage([box[0], box[1]], [box[2], box[3]], img)

# Helper function to check the gradient by using centered finite differences and space step epsilon
def check_gradient(f,gradf,x0):
    N = len(x0)
    g  = gradf(x0)
    gg = np.zeros_like(x0)
    for i in range(N):
        eps = 1e-4
        e = np.zeros(N)
        e[i] = eps
        gg[i] = (f(x0+e) - f(x0-e))/(2*eps)
    err = abs(gg-g)
    print(' gradient error {:.3e} (L2), {:.3e} (min), {:.3e} (max)'.format(np.linalg.norm(err), err.min(), err.max()))


if __name__ == '__main__':
    N = 10 # number of diracs
    n = 10 # number of pixels in each direction (image)

    print('First example:  uniform density on unit square...', end=' ', flush=True)
    dens = make_square(box=[0,0,1,1])
    masses = 1.0/N*np.ones(N)
    check_gradient(lambda x: flat_wass(dens, x, masses)[0],
                   lambda x: flat_wass(dens, x, masses)[1],
                   5*np.random.rand(2*N))

    print('Second example: uniform density on a rectangle...', end=' ', flush=True)
    # here do not forget to rescale weights by density measure
    dens = make_square(box=[1,0,3,1])
    masses = dens.measure()/N*np.ones(N)
    check_gradient(lambda x: flat_wass(dens, x, masses)[0],
                   lambda x: flat_wass(dens, x, masses)[1],
                   5*np.random.rand(2*N))


    print('Third example:  density based on a given image...', end=' ', flush=True)
    # we first create data on a Cartesian grid (this will represent our 5x5 pixels image)
    t = np.linspace(-1.0,1.0,n)
    x, y = t[None,:], t[:,None]
    img = np.exp(-2*(x**2 + y**2))

    # we can create the density by using the make_image utility function
    dens   = make_image(img, box=[0,0,1,1])
    masses = dens.measure()/N*np.ones(N)
    np.random.seed(0)
    check_gradient(lambda x: flat_wass(dens, x, masses)[0],
                   lambda x: flat_wass(dens, x, masses)[1],
                   5*np.random.rand(2*N))
    
    print('Fourth example: minimize Wasserstein distance over dirac locations...')
    # Minimize flat_wass(dens,.) where dens is a 128x128 image of a Gaussian, over N=1024 diracs
    from scipy.optimize import minimize
    pd_path='/tmp/power_diagram.vtk'

    n = 128
    t = np.linspace(0,1,n)
    x, y = t[None,:], t[:,None]
    img = (np.cos(2*np.pi*x)*np.cos(2*np.pi*y))**2
    dens = make_image(img, box=[0,0,1,1])

    N = 1024
    Y0 = (np.random.rand(2*N)-0.5)*2
    masses = (dens.measure()/N)*np.ones(N)
    
    print('  > distance: ', end='', flush=True)
    phi = lambda Y: flat_wass(dens, Y, masses)
    log = lambda Y: print('{:.8e}'.format(phi(Y)[0]), end=', ', flush=True)
    sol = minimize(phi, jac=True, x0=Y0, method='CG', callback=log)
    Yopt = sol.x

    print('{:.8e}\n  > optimization terminated with status {}.'.format(phi(Yopt)[0], sol.status))
    if not sol.success:
        msg='Optimization did not succeed because: {}.'.format(sol.message)
        raise RuntimeError(msg)
    
    print('  > dumping resulting power diagram to \'file://{}\'...'.format(pd_path))
    pd = optimal_transport(dens, Yopt.reshape(-1,2), masses)
    pd.display_vtk(pd_path)
