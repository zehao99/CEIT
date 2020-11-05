import matplotlib.pyplot as plt
import numpy as np

from MyEIT.EITPlotter import EITPlotter
from MyEIT.Solver import Solver

solver = Solver()
plotter = EITPlotter()
delta_V = np.random.rand(240)

fig, ax = plt.subplots(nrows=1, ncols=1)
plotter.plot_detection_area_map(solver.solve(delta_V), ax)
plt.show()
