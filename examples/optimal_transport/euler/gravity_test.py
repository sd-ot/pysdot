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


def update_positions_to_get_centroids(ot, b_obj):
    nb_diracs = ot.nb_diracs()
    dim = ot.dim()

    #ot.set_positions(b_obj)
    for sub_iter in range(50):
        mvs = ot.pd.der_centroids_and_integrals_wrt_weight_and_positions()
        m = csr_matrix((mvs.m_values, mvs.m_columns, mvs.m_offsets))

        rd = np.arange(dim * nb_diracs, dtype=np.int)
        b0 = (dim + 1) * np.floor_divide(rd, dim)
        l0 = b0 + rd % dim
        l1 = (dim + 1) * np.arange(nb_diracs, dtype=np.int) + dim
        C = m[l0, :][:, l0]
        D = m[l0, :][:, l1]
        E = m[l1, :][:, l0]
        F = m[l1, :][:, l1]

        # print(np.linalg.cond(F.todense()))

        G = C - D * spsolve(F.tocsc(), E.tocsc())
        b = np.array(b_obj.flat) - mvs.v_values[l0]
        # print(np.real(eigvals((np.transpose(G) * G).todense())))
        # print(np.linalg.cond((G*G).todense()))
        # print(ot.get_positions())
        # print(ot.get_weights())
        # pm(m)
        # pm(C)
        # pm(D)
        # pm(E)
        # pm(F)
        # print(np.real(eigvals((np.transpose(G) * G).todense())))
        # ly = np.arange(nb_diracs, dtype=np.int) * 2 + 1
        # H = G[ly, :][:, ly]
        # print(eig((np.transpose(H) * H).todense()))
        # pm(H)
        # break
        p = 1e-2 * np.max(G)
        M = np.transpose(G) * G + p * diag(dim * nb_diracs)
        V = np.transpose(G) * b # + p * np.ones(dim * nb_diracs)

        X = spsolve(M, V)
        print(sub_iter, np.linalg.norm(X))
        m = 5e-2
        if np.linalg.norm(X) > m:
            X *= m / np.linalg.norm(X)

        ot.set_positions(ot.get_positions() + 0.8 * X.reshape((-1, dim)))
        ot.adjust_weights()


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
            for x in np.linspace(radius, l - radius, n):
                nx = x  # + 0.2 * radius * (np.random.rand() - 0.5)
                ny = y  # + 0.2 * radius * (np.random.rand() - 0.5)
                positions.append([nx, ny])
    positions = np.array(positions)
    nb_diracs = positions.shape[0]
    dim = positions.shape[1]

    # OptimalTransport
    ot = OptimalTransport(domain, RadialFuncInBall())
    ot.set_weights(np.ones(nb_diracs) * radius**2)
    ot.set_masses(np.ones(nb_diracs) * mass)
    ot.set_positions(positions)
    ot.adjust_weights()

    ot.display_vtk(base_filename + "0.vtk", points=True)

    g = np.zeros((nb_diracs, dim))
    g[:, 1] = -0.001

    bh = [ot.get_centroids()]  # history of centroids
    for num_iter in range(50):
        bh.append(ot.get_centroids())
        print("num_iter", num_iter)

        # proposition for centroids
        bn = 2 * bh[-1] - bh[-2] + g

        # find a new set of diracs parameters (position and weight)
        # to be as close to the new centroids as possible
        update_positions_to_get_centroids(ot, bn)

        # display
        n1 = num_iter + 1
        ot.display_vtk(base_filename + "{}.vtk".format(n1), points=True)
        # ot.pd.display_vtk_points(base_filename + "pts_{}.vtk".format(n1))


os.system("rm results/pd_*")
run(5, "results/pd_")
