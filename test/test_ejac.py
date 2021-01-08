from unittest import TestCase
from CEIT.EJAC import EJAC
from CEIT.readmesh import read_mesh_from_csv
import os


class TestEJAC(TestCase):

    def test_calc_detection_elements(self):
        self.EJAC = EJAC()
