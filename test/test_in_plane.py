from unittest import TestCase
from locate_points import locate_points
from pytest import approx

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

    # def test_distance(self):
    #     assert(False)

    # def test_inside(self):
    #     assert(True)

    # def test_outside(self):
    #     assert(True)

    # def test_orientation_sensitivity(self):
    #     assert(True)
