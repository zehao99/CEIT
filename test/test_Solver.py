from unittest import TestCase
from CEIT.Solver import reinitialize_solver
from CEIT.EJAC import EJAC

class TestSolver(TestCase):
    def test_reinitialize_solver(self):
        self.EJAC = EJAC()
        self.EJAC.save_JAC_np()
        reinitialize_solver(203)
        # pass