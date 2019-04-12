from .radial_funcs import RadialFuncUnit
from .cpp import cpp_module
import numpy as np
import os


class PowerDiagram:
    def __init__(self, domain=None, radial_func=RadialFuncUnit()):
        self.radial_func = radial_func
        self.domain = domain

        self.positions = None
        self.weights = None

        self._positions_are_new = True
        self._weights_are_new = True
        self._domain_is_new = True
        self._inst = None

    def set_positions(self, positions):
        self._positions_are_new = True
        self.positions = positions

    def set_weights(self, weights):
        self._weights_are_new = True
        self.weights = weights

    def set_domain(self, domain):
        self._domain_is_new = True
        self.domain = domain

    def integrals(self):
        inst = self._updated_grid()
        return inst.integrals(
            self.positions,
            self.weights,
            self.domain._inst,
            self.radial_func.name()
        )

    def centroids(self):
        inst = self._updated_grid()
        return inst.centroids(
            self.positions,
            self.weights,
            self.domain._inst,
            self.radial_func.name()
        )

    def der_integrals_wrt_weights(self):
        inst = self._updated_grid()
        return inst.der_integrals_wrt_weights(
            self.positions,
            self.weights,
            self.domain._inst,
            self.radial_func.name()
        )

    def der_centroids_and_integrals_wrt_weight_and_positions(self):
        inst = self._updated_grid()
        return inst.der_centroids_and_integrals_wrt_weight_and_positions(
            self.positions,
            self.weights,
            self.domain._inst,
            self.radial_func.name()
        )

    def display_vtk(self, filename, points=False, centroids=False):
        dn = os.path.dirname(filename)
        if len(dn):
            os.makedirs(dn, exist_ok=True)
        inst = self._updated_grid()
        return inst.display_vtk(
            self.positions,
            self.weights,
            self.domain._inst,
            self.radial_func.name(),
            filename,
            points,
            centroids
        )

    def display_vtk_points(self, filename, points=None):
        if points is None:
            return self.display_vtk_points(filename, self.positions)
        dn = os.path.dirname(filename)
        if len(dn):
            os.makedirs(dn, exist_ok=True)
        inst = self._updated_grid()
        return inst.display_vtk_points(
            self.positions,
            filename
        )

    # make a .asy file for a representation of the power diagram
    def display_asy(self, filename, preamble="", closing="", output_format="pdf", linewidth=0.02, dotwidth=0.0, values=np.array([]), colormap="inferno", avoid_bounds=False, min_rf=1, max_rf=0):
        dn = os.path.dirname( filename )
        if len( dn ):
            os.makedirs( dn, exist_ok = True )

        p = "settings.outformat = \"{}\";\nunitsize(1cm);\n".format( output_format )
        if linewidth > 0:
            p += "defaultpen({}cm);\n".format( linewidth )
        elif dotwidth > 0:
            p += "defaultpen({}cm);\n".format( dotwidth / 6 )
        p += preamble

        inst = self._updated_grid()
        inst.display_asy(
            self.positions,
            self.weights,
            self.domain._inst,
            self.radial_func.name(),
            filename,
            p,
            values,
            colormap,
            linewidth,
            dotwidth,
            avoid_bounds,
            closing,
            min_rf,
            max_rf
        )


    def _updated_grid(self):
        # check types
        if not isinstance(self.positions, np.ndarray):
            self.positions = np.array(self.positions)
        if not isinstance(self.weights, np.ndarray):
            self.weights = np.array(self.weights)

        # instantiation of PowerDiagram
        if not self._inst:
            assert(self.positions.dtype == self.domain._type)
            assert(self.weights.dtype == self.domain._type)
            module = cpp_module.module_for_type_and_dim(
                self.domain._type, self.positions.shape[1]
            )
            self._inst = module.PowerDiagramZGrid(11)

        self._inst.update(
            self.positions,
            self.weights,
            self._positions_are_new or self._domain_is_new,
            self._weights_are_new or self._domain_is_new,
            self.radial_func.name()
        )
        self._positions_are_new = False
        self._weights_are_new = False
        self._domain_is_new = False

        return self._inst
        
    # def boundary_integral( self ):
    #     inst = self._updated_grid()
    #     return inst.boundary_integral(
    #         self.positions,
    #         self.weights,
    #         self.domain._inst,
    #         self.radial_func.name()
    #     )

    # def der_boundary_integral( self ):
    #     inst = self._updated_grid()
    #     return inst.der_boundary_integral(
    #         self.positions,
    #         self.weights,
    #         self.domain._inst,
    #         self.radial_func.name()
    #     )
