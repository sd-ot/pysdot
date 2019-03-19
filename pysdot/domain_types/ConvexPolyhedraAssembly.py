from ..cpp import cpp_module
import numpy as np
import os


#
class ConvexPolyhedraAssembly:
    def __init__(self, type=np.float64):
        """
            type => scalar type used to store coordinates, compute volumes...
        """
        self._inst = None
        self._type = type

    def add_box(self, min_pos, max_pos, coeff=1.0, cut_id=-1):
        a_min_pos = np.array(min_pos, dtype=self._type)
        a_max_pos = np.array(max_pos, dtype=self._type)
        inst = self._update_inst([a_min_pos.shape[0], a_max_pos.shape[0]])
        inst.add_box(
            a_min_pos,
            a_max_pos,
            self._type(coeff),
            np.uint64(cut_id)
        )

    def measure(self):
        if self._inst:
            return self._inst.measure()
        return self._type(0)

    def _update_inst(self, dimensions):
        if self._inst:
            return
        
        for i in range(1, len(dimensions)):
            assert(dimensions[i] == dimensions[0])

        module = cpp_module.module_for_type_and_dim(self._type, dimensions[0])
        self._inst = module.ConvexPolyhedraAssembly()
        return self._inst

    # def sub_box(self, min_pos, max_pos, coeff=1.0, cut_id=-1):
    #     self.add_box( min_pos, max_pos, - coeff, cut_id )

    # # remember to call normalize when integration( coeff ) != 1
    # def add_convex_polyhedron( self, positions_and_normals, coeff=1.0, cut_id=-1 ):
    #     pan = np.array( positions_and_normals, dtype=self._type )
    #     if len( pan.shape ) == 1:
    #         pan = pan.reshape( -1, 2 * self._dim )
    #     self._inst.add_convex_polyhedron( pan, self._type( coeff ), np.uint64( cut_id ) )

    # def normalize( self ):
    #     self._inst.normalize()

    # def display_boundaries_vtk( self, filename ):
    #     os.makedirs( os.path.dirname( filename ), exist_ok = True )
    #     self._inst.display_boundaries_vtk( filename )

    # # coefficient at `point`. If point is not contained, return 0.
    # def coeff_at( self, point ):
    #     return self._inst.coeff_at( np.array( point, dtype=self._type ) )

    # # 
    # def min_position( self ):
    #     return self._inst.min_position()

    # # 
    # def max_position( self ):
    #     return self._inst.max_position()

    # # True if points is contained
    # def contains( self, point ):
    #     return self.coeff_at( point ) != 0
