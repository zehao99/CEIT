from MyEIT.solver import Solver
import numpy as np

solver = Solver()

delta_V = np.random.rand(240)

solver.solve(delta_V)