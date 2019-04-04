from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from pysdot import OptimalTransport

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.linalg import eigvals
from scipy.linalg import eig
import matplotlib.pyplot as plt
import scipy.optimize
import numpy as np
import scipy
import os


class Optimizer:
    def __init__(self, func):
        self.func = func

    def run(self, x):
        scipy.optimize.minimize(
            self.func,
            x,
            (),
            # jac=jac,
            tol=1e-6,
            method='BFGS',
            # options={'disp': True}
        )


class System:
    def __init__(self, domain, positions, velocity, masses, base_filename, dt=1.0):
        self.position_history = [
            np.array(positions) - dt * velocity,
            np.array(positions)
        ]
        self.centroid_history = []
        self.weight_history = [np.array(masses)/np.pi]
        self.mass_history = [np.array(masses)]

        self.base_filename = base_filename
        self.domain = domain
        self.dt = dt

        self.cpt_display = 0

    def get_new_weights_for_areas(self):
        ot = self.opt_trans()
        ot.update_weights()
        self.weight_history.append(ot.get_weights())
        self.centroid_history.append(ot.get_centroids())

    def get_new_positions_for_centroids(self):
        scipy.optimize.minimize(
            self.obj,
            self.position_history[-1].flatten(),
            (),
            # jac=jac,
            tol=1e-6,
            method='BFGS',
            # options={'disp': True}
        )

    def obj(self, x, ret_jac=False):
        ot = self.opt_trans()
        ot.set_positions(x.reshape((-1, 2)))

        # current centroids
        current_centroids = ot.get_centroids()

        # target centroids
        oc = self.centroid_history[-1]
        target_centroids = oc + self.velocity * self.dt

        dlt = current_centroids - target_centroids
        return np.sum(dlt**2)

    def opt_trans(self):
        ot = OptimalTransport(self.domain, RadialFuncInBall())
        ot.set_positions(self.position_history[-1])
        ot.set_weights(self.weight_history[-1])
        ot.set_masses(self.mass_history[-1])
        return ot

    def display(self):
        ot = self.opt_trans()

        fn = "{}{}.vtk".format(self.base_filename, self.cpt_display)
        ot.display_vtk(fn, points=True)
        self.cpt_display += 1



def run(n, base_filename, l=0.5):
    os.system("rm {}*.vtk".format(base_filename))

    domain = ConvexPolyhedraAssembly()
    domain.add_box([0, 0], [1, 1])

    positions = []
    velocity = []
    radius = 0.5 * l / n
    for y in np.linspace(radius, l - radius, n):
        for x in np.linspace(0.5 - l / 2 + radius, 0.5 + l / 2 - radius, n):
            nx = x + 0.5 * radius * (np.random.rand() - 0.5)
            ny = y + 0.5 * radius * (np.random.rand() - 0.5)
            velocities.append([0, -radius/5])
            positions.append([nx, ny])

    masses = np.ones(len(positions)) * np.pi * radius**2
    s = System(domain, positions, velocities, masses, base_filename)
    s.get_new_weights_for_areas()
    s.display()

    for num_iter in range(10):
        print(num_iter)

        s.get_new_positions_for_centroids()
        s.get_new_weights_for_areas()
        s.display()

#
run(10, "results/pd_")



# def obj(cx, ot, bh, dt):
#     pc = cx.reshape((-1, 2))
#     ot.set_positions(pc)
#     ot.update_weights()

#     bm = np.array(bh[-2].flat)
#     b0 = np.array(bh[-1].flat)
#     bc = np.array(ot.get_centroids().flat)
#     bt = 2 * b0 - bm

#     dlt = bc - bt
#     dlp = cx - bt
#     return 0.5 * np.sum(dlt**2) \
#         + 0.5 * 1e-4 * np.sum(dlp**2)


# def jac(cx, ot, bh, dt):
#     nb_diracs = ot.nb_diracs()
#     dim = ot.dim()

#     # get G
#     mvs = ot.pd.der_centroids_and_integrals_wrt_weight_and_positions()
#     # if mvs.error:
#     m = csr_matrix((mvs.m_values, mvs.m_columns, mvs.m_offsets))

#     rd = np.arange(dim * nb_diracs, dtype=np.int)
#     b0 = (dim + 1) * np.floor_divide(rd, dim)
#     l0 = b0 + rd % dim
#     C = m[l0, :][:, l0]

#     # centroids
#     pc = cx.reshape((-1, 2))
#     ot.set_positions(pc)

#     bm = np.array(bh[-2].flat)
#     b0 = np.array(bh[-1].flat)
#     bc = np.array(ot.get_centroids().flat)
#     bt = 2 * b0 - bm

#     dlt = bc - bt
#     # dlp = cx - bt
#     return C.transpose() * dlt  # + 1e-2 * bt


# def get_new_positions_for_centroids(ot, bh, dt):
#     ropt = scipy.optimize.minimize(
#         obj,
#         ot.get_positions().flatten(),
#         (ot, bh, dt),
#         jac=jac,
#         tol=1e-6,
#         method='CG',
#         # options={'disp': True}
#     )
#     # print(ropt.njev)

#     positions = ropt.x.reshape((-1, 2))
#     ot.set_positions(positions)


# def run(n, base_filename, l=0.5):
#     # domain
#     domain = ConvexPolyhedraAssembly()
#     domain.add_box([0, 0], [1, 1])

#     # initial positions, weights and masses
#     positions = []
#     if n == 1:
#         radius = 0.3
#         mass = 3.14159 * radius**2
#         positions.append([0.5, radius])
#     else:
#         radius = l / (2 * (n - 1))
#         mass = l**2 / n**2
#         for y in np.linspace(radius, l - radius, n):
#             for x in np.linspace(0.5 - l / 2 + radius, 0.5 + l / 2 - radius, n):
#                 nx = x + 0.5 * radius * (np.random.rand() - 0.5)
#                 ny = y + radius + 0.5 * radius * (np.random.rand() - 0.5)
#                 positions.append([nx, ny])
#     positions = np.array(positions)
#     nb_diracs = positions.shape[0]

#     # OptimalTransport
#     ot = OptimalTransport(domain, RadialFuncInBall())
#     ot.set_weights(np.ones(nb_diracs) * radius**2)
#     ot.set_masses(np.ones(nb_diracs) * mass)
#     ot.set_positions(positions)
#     ot.update_weights()

#     ot.display_vtk(base_filename + "0.vtk", points=True)

#     # history of centroids
#     ce = ot.get_centroids()
#     ce[:, 1] += radius / 10
#     bh = [ce]

#     dt = 1.0
#     for num_iter in range(500):
#         print("num_iter", num_iter)

#         bh.append(ot.get_centroids())
#         get_new_positions_for_centroids(ot, bh, dt)
#         ot.update_weights()

#         # display
#         n1 = int(num_iter / 1) + 1
#         ot.display_vtk(base_filename + "{}.vtk".format(n1), points=True)


# os.system("rm results/pd_*")
# run(5, "results/pd_")
