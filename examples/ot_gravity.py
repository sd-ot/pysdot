from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from pysdot import OptimalTransport

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.linalg import eigvals
from scipy.linalg import eig
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os


def pm(G):
    print(np.array2string(G.todense(), 5000))


def diag(n):
    return scipy.sparse.diags([np.ones(n)], [0])


def update_positions(ot, bh, dt):
    """
    Pb: la determination des poids n'est pas assez solide.
    """
    dim = ot.dim()

    ratio = 1.0
    last_change = 0
    old_X = ot.get_positions() + 0.0
    old_w = ot.get_weights() + 0.0
    os.system("rm results/sub_iter_*")
    for sub_iter in range(100):
        if sub_iter == last_change + 300 / ratio:
            last_change = sub_iter
            ot.set_positions(old_X)
            ot.set_weights(old_w)
            ratio *= 0.5
            print("c", ratio)
            continue

        #
        # ot.coalesce_close_diracs(1e-5, bh)
        nb_diracs = ot.nb_diracs()

        ot.display_vtk("results/sub_iter_{}.vtk".format(sub_iter), points=True)
        # print(ot.nb_diracs(),bh[-1].shape)

        # g = np.zeros((nb_diracs, dim))
        # g[:, 1] = -0.001

        # get G
        mvs = ot.pd.der_centroids_and_integrals_wrt_weight_and_positions()
        if mvs.error:
            last_change = sub_iter
            ot.set_positions(old_X)
            ot.set_weights(old_w)
            ratio *= 0.5
            print("r", ratio)
            continue
        m = csr_matrix((mvs.m_values, mvs.m_columns, mvs.m_offsets))

        rd = np.arange(dim * nb_diracs, dtype=np.int)
        b0 = (dim + 1) * np.floor_divide(rd, dim)
        l0 = b0 + rd % dim
        l1 = (dim + 1) * np.arange(nb_diracs, dtype=np.int) + dim
        C = m[l0, :][:, l0]
        D = m[l0, :][:, l1]
        E = m[l1, :][:, l0]
        F = m[l1, :][:, l1]

        G = C - D * spsolve(F.tocsc(), E.tocsc())

        # centroids
        bm = np.array(bh[-2].flat)
        b0 = np.array(bh[-1].flat)
        b1 = mvs.v_values[l0]

        # db = b1 - np.array(ot.get_positions().flat)

        # system to be solved
        p = 1e-5 * np.max(G)
        M = np.transpose(G) * G + p * diag(dim * nb_diracs)
        V = np.transpose(G) * (2 * b0 - bm - b1) # + p * db

        # solve it
        m = 5e-2
        X = spsolve(M, V)
        n = np.linalg.norm(X)
        print(sub_iter, n)
        if n > m:
            X *= m / n

        ot.set_positions(ot.get_positions() + ratio * X.reshape((-1, dim)))
        if ot.update_weights(True):
            last_change = sub_iter
            ot.set_positions(old_X)
            ot.set_weights(old_w)
            ratio *= 0.5
            print("s", ratio)
            continue

        if n < 1e-5:
            break


def run(n, base_filename, l=0.5):
    # domain
    domain = ConvexPolyhedraAssembly()
    domain.add_box([0, 0], [1, 1])

    # initial positions, weights and masses
    positions = []
    if n == 1:
        radius = 0.3
        mass = 3.14159 * radius**2
        positions.append([0.5, radius])
    else:
        radius = l / (2 * (n - 1))
        mass = l**2 / n**2
        for y in np.linspace(radius, l - radius, n):
            for x in np.linspace(0.5 - l / 2 + radius, 0.5 + l / 2 - radius, n):
                nx = x + 0.5 * radius * (np.random.rand() - 0.5)
                ny = y + radius + 0.5 * radius * (np.random.rand() - 0.5)
                positions.append([nx, ny])
    positions = np.array(positions)
    nb_diracs = positions.shape[0]

    # OptimalTransport
    ot = OptimalTransport(domain, RadialFuncInBall())
    ot.set_weights(np.ones(nb_diracs) * radius**2)
    ot.set_masses(np.ones(nb_diracs) * mass)
    ot.set_positions(positions)
    ot.update_weights()

    ot.display_vtk(base_filename + "0.vtk", points=True)

    # history of centroids
    ce = ot.get_centroids()
    ce[:, 1] += radius / 10
    bh = [ce]

    dt = 1.0
    for num_iter in range(500):
        bh.append(ot.get_centroids())
        print("num_iter", num_iter)

        update_positions(ot, bh, dt)

        # display
        n1 = int(num_iter / 1) + 1
        ot.display_vtk(base_filename + "{}.vtk".format(n1), points=True)


os.system("rm results/pd_*")
run(15, "results/pd_")
