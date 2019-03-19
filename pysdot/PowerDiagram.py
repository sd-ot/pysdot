from .radial_funcs import RadialFuncUnit
from .cpp import cpp_module
import numpy as np


class PowerDiagram:
    def __init__(self, domain=None, radial_func=RadialFuncUnit()):
        self.positions_are_new = True
        self.weights_are_new = True
        self.domain_is_new = True

        self.radial_func = radial_func
        self._inst = None

        if domain:
            self.set_domain(domain)

    def set_positions(self, positions):
        self.positions_are_new = True
        self.positions = positions

    def set_weights(self, weights):
        self.weights_are_new = True
        self.weights = weights

    def set_domain(self, domain):
        self.domain_is_new = True
        self.domain = domain

    def integrals(self):
        inst = self.update_if_necessary()
        return inst.integrals(
            self.positions,
            self.weights,
            self.domain._inst,
            self.radial_func.name()
        )

    def der_integrals_wrt_weights(self):
        inst = self.update_if_necessary()
        return inst.der_integrals_wrt_weights(
            self.positions,
            self.weights,
            self.domain._inst,
            self.radial_func.name()
        )

    def centroids(self):
        inst = self.update_if_necessary()
        return inst.centroids(
            self.positions,
            self.weights,
            self.domain._inst,
            self.radial_func.name()
        )

    def update_if_necessary(self):
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
            self.positions_are_new or self.domain_is_new,
            self.weights_are_new or self.domain_is_new,
            self.radial_func.name()
        )
        self.positions_are_new = False
        self.weights_are_new = False
        self.domain_is_new = False

        return self._inst
