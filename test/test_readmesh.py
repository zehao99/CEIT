from unittest import TestCase
from CEIT.readmesh import ReadMesh, init_mesh


class TestReadMesh(TestCase):
    def test_return_mesh(self):
        mesh_obj, electrode_num, electrode_centers, electrode_radius = init_mesh()
        if mesh_obj is None:
            self.fail()
        else:
            self.assertEqual(True, True)
