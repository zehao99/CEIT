from unittest import TestCase
from MyEIT.Solver import reinitialize_solver

class Test(TestCase):
    def test_reinitialize_solver(self):
        reinitialize_solver(203)
