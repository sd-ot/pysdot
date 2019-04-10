from pysdot.radial_funcs import RadialFuncInBall
from pysdot import OptimalTransport

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.linalg import eigvals
import numpy as np
import scipy
import sys

# import matplotlib.pyplot as plt


def diag(n):
    return scipy.sparse.diags([np.ones(n)], [0])


class FluidSystem:
    def __init__( self, domain, positions, velocities, masses, base_filename ):
        self.ot = OptimalTransport(domain, RadialFuncInBall())
        self.ot.set_positions(np.array(positions))
        self.ot.set_weights(np.array(masses)/np.pi)
        self.ot.set_masses(np.array(masses))

        self.base_filename = base_filename
        self.cpt_display = 0
        self.max_iter = 200
        self.time = 0

        # initial centroid positions and velocities
        self.ot.update_weights()
        self.centroids = self.ot.get_centroids()
        self.velocities = np.array(velocities)
        self.coeff_centroid_force = 1e-4

    def display( self ):
        fn = "{}{}.vtk".format( self.base_filename, self.cpt_display )
        self.ot.display_vtk( fn, points=True, centroids=True )
        self.cpt_display += 1

    def make_step( self ):
        ratio_dt = 1.0
        while self.try_step( ratio_dt ) == False:
            ratio_dt *= 0.5
            print( "  dt ratio:", ratio_dt )

    def try_step( self, ratio_dt ):
        old_p = self.ot.get_positions()

        # find dt
        radii_ap = ( np.array( self.ot.get_masses() ) / np.pi ) ** 0.5
        vn2 = np.linalg.norm( self.velocities, axis=1, ord=2 )
        dt = ratio_dt * 0.2 / np.max( np.abs( vn2 / radii_ap ) )
        adv = dt * self.velocities

        # target centroid positions + initial guess for the dirac positions
        target_centroids = self.centroids + adv
        self.ot.set_positions( old_p + adv )

        # stuff to extract centroids, masses, ...
        d = self.ot.dim()
        n = self.ot.nb_diracs()
        rd = np.arange( d * n, dtype=np.int )
        b0 = ( d + 1 ) * np.floor_divide( rd, d )
        l0 = b0 + rd % d # l1 = (d + 1) * np.arange(n, dtype=np.int) + d

        # find positions to fit the target centroid positions
        ratio = 1.0
        for num_iter in range( self.max_iter + 1 ):
            if num_iter == self.max_iter:
                self.ot.set_positions( old_p )
                return False

            # search dir
            mvs = self.ot.pd.der_centroids_and_integrals_wrt_weight_and_positions()
            if mvs.error:
                self.ot.set_positions( old_p )
                ratio *= 0.5
                if ratio < 1e-2:
                    return False
                print( "  solve X ratio:", ratio )
                continue

            M = csr_matrix( ( mvs.m_values, mvs.m_columns, mvs.m_offsets ) )[ l0, : ][ :, l0 ]
            V = mvs.v_values[ l0 ] - target_centroids.flatten()

            c = self.coeff_centroid_force * np.max( M )
            V += c * ( self.ot.get_positions() - target_centroids ).flatten()
            M += c * diag( 2 * n )

            X = spsolve( M, V ).reshape( ( -1, d ) )
            # if np.linalg.norm( X, ord=np.inf ) > self.max_disp_at_each_sub_iter:
            #     X *= self.max_disp_at_each_sub_iter / np.linalg.norm( X, ord=np.inf )

            self.ot.set_positions( self.ot.get_positions() - ratio * X )

            e = np.linalg.norm( X )
            # print( "  e", e )
            if e < 1e-6:
                break

        # projection
        # self.ot.verbosity = 1
        self.ot.update_weights( relax=0.75 )

        # update centroid pos and speed
        self.time += dt
        old_centroids = self.centroids
        self.centroids = self.ot.get_centroids()
        self.velocities = ( self.centroids - old_centroids ) / dt
        
        return True