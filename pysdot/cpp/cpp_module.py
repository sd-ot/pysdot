import pybind_sdot_2d_double
import pybind_sdot_3d_double
import numpy


def module_for_type_and_dim(type, dim):
    if type == numpy.float64:
        if dim == 2:
            return pybind_sdot_2d_double
        if dim == 3:
            return pybind_sdot_3d_double
    return None
