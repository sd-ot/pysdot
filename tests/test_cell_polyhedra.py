import numpy as np

from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import PowerDiagram

def test_2D():
    pd = PowerDiagram( [ [ 0.3, 0.0 ], [ 0.7, 0.0 ] ])
    offset, coordinates = pd.cell_polyhedra()
    for num_cell in range( pd.positions.shape[ 0 ] ):
        print( coordinates[ offset[ num_cell + 0 ] : offset[ num_cell + 1 ] ] )

def test_3D():
    pd = PowerDiagram( [ [ 0.3, 0.0, 0.0 ], [ 0.7, 0.0, 0.0 ] ])
    offset_polyhedra, offset_polygons, coordinates = pd.cell_polyhedra()
    for num_cell in range( pd.positions.shape[ 0 ] ):
        print( "num_cell", num_cell )
        for i in range( offset_polyhedra[ num_cell + 0 ], offset_polyhedra[ num_cell + 1 ] ):
            print( coordinates[ offset_polygons[ i + 0 ] : offset_polygons[ i + 1 ] ] )

test_3D()
