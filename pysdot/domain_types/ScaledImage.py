from ..cpp import cpp_module
import numpy as np
import os


#
class ScaledImage:
    def __init__(self, min_pt, max_pt, img):
        module = cpp_module.module_for_type_and_dim( np.float64, len( img.shape ) )
        self._inst = module.ScaledImage( min_pt, max_pt, img )
        self._type = np.float64

    def measure(self):
        return self._inst.measure()

    # def normalize( self ):
    #     self._inst.normalize()

    def display_boundaries_vtk( self, filename ):
        os.makedirs( os.path.dirname( filename ), exist_ok = True )
        self._inst.display_boundaries_vtk( filename )

    # coefficient at `point`. If point is not contained, return 0.
    def coeff_at( self, point ):
        return self._inst.coeff_at( np.array( point ) )

    # 
    def min_position( self ):
        return self._inst.min_position()

    # 
    def max_position( self ):
        return self._inst.max_position()

    # True if points is contained
    def contains( self, point ):
        return self.coeff_at( point ) != 0

    def _update_inst(self, dimensions):
        return self._inst
