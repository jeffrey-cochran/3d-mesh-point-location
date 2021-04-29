from unittest import TestCase
from locate_points import locate_points
import cupy as cp
from numpy import loadtxt, allclose
from pytest import approx
from constants import query_dir, test_data_dir
from os.path import join

class tetrahedron(TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return

    def test_coplanar(self):
        results = locate_points(
            mesh_prefix="test_tetra",
            pts_prefix="test_tetra_coplanar",
            chunk_size=2
        )
        assert(
            approx(results[1].get().sum(),  0.)
        )

    def test_corners(self):
        results = locate_points(
            mesh_prefix="test_tetra",
            pts_prefix="test_tetra_corners",
            chunk_size=4
        )
        assert(
            approx(results[1].get().sum(),  0.)
        )
        assert( 
            approx( 
                results[2].get(),
                cp.asarray([
                    [0.,  0.,  0.],
                    [0., -5.,  0.],
                    [0.,  0.,  5.],
                    [5.,  0.,  0.],
                ])
            )
        )

    def test_edges(self):
        results = locate_points(
            mesh_prefix="test_tetra",
            pts_prefix="test_tetra_edges",
            chunk_size=5
        )
        assert(
            approx(results[1].get().sum(),  0.)
        )
        assert( 
            approx( 
                results[2].get(),
                cp.asarray([
                    [0.,  -2.5,  0.],
                    [0.,    0.,  2.5],
                    [2.5,   0.,  0.],
                    [0., -2.5,  2.5],
                    [2.5, -2.5,  0.]
                ])
            )
        )
    
    def test_distance(self):
        results = locate_points(
            mesh_prefix="test_tetra",
            pts_prefix="test_tetra_distance",
            chunk_size=5
        )
        
        assert(
            approx(results[1].get().sum(),  0.)
        )
        assert( 
            approx( 
                results[1].get(),
                cp.asarray(
                    [0.1,  -0.1,  0.1, -0.1, 0.1, -0.1, 0.1, -0.1]
                    )
            )
        )

    def test_inside(self):
        results = locate_points(
            mesh_prefix="test_tetra",
            pts_prefix="test_tetra_inside",
            chunk_size=5
        )
        assert( 
            cp.less_equal(
                results[1],
                0.
            ).all()
        )

    def test_outside(self):
        results = locate_points(
            mesh_prefix="test_tetra",
            pts_prefix="test_tetra_outside",
            chunk_size=100
        )
        assert( 
            cp.greater_equal(
                results[1],
                0.
            ).all()
        )   

    def test_big_mesh(self):
        bounding_box = {
            'rotation_matrix': [
                [0.77498806, -0.56608065, -0.28097367],
                [0.58471266,  0.81096618, -0.02109427],
                [0.2398012,  -0.14794105,  0.95948363]
            ],
            'translation_vector': [44.73739702, -192.96706105, 634.59906255],
            'size': [3.95018717, 14.12505279, 3.52829708]
        }

        results = locate_points(
            mesh_prefix="test_big_mesh",
            pts_prefix="test_big_mesh",
            chunk_size=100,
            bounding_box=bounding_box
        )
        test_distances = results[1].get()
        validation_distances = loadtxt(join(test_data_dir, "test_big_mesh_signed_distances.csv"))
        assert(allclose(test_distances, validation_distances))

    def test_bounding_box_inside(self):
        bounding_box = {
            'rotation_matrix': [
                [1.,  0., 0.],
                [0., -1., 0.],
                [0.,  0., 1.]
            ],
            'translation_vector': [0., 0., 0.],
            'size': [1., 2., 2.]
        }

        results = locate_points(
            mesh_prefix="test_tetra",
            pts_prefix="test_tetra_bbox_inside",
            chunk_size=5,
            bounding_box=bounding_box
        )

        assert( cp.less_equal(results[1], 0.).all() )

    def test_bounding_box_outside(self):
        bounding_box = {
            'rotation_matrix': [
                [1.,  0., 0.],
                [0., -1., 0.],
                [0.,  0., 1.]
            ],
            'translation_vector': [0., 0., 0.],
            'size': [1., 1., 1.]
        }

        results = locate_points(
            mesh_prefix="test_tetra",
            pts_prefix="test_tetra_bbox_outside",
            chunk_size=5,
            bounding_box=bounding_box
        )

        assert( cp.greater_equal(results[1], 0.).all() )