from .EFEM import EFEM
from .models.mesh import MeshObj
from .util.utilities import get_config
import numpy as np


class Simulator:
    def __init__(self, mesh=None):
        if mesh is None:
            self.mesh = MeshObj()
        else:
            self.mesh = mesh
        self.config = get_config()
        self.fwd_simulator = EFEM(self.mesh)
        self.electrode_potentials = np.zeros(
            self.mesh.electrode_num * (self.mesh.electrode_num - 1))

    def calc_potential(self):
        """
        Calculate the potential according to the current parameter state.

        Returns: Array of potential amplitudes.

        """
        for i in range(self.mesh.electrode_num):
            cnt = 0
            _, _, electrode_potential = self.fwd_simulator.calculation(i)
            for j, v in enumerate(electrode_potential):
                if j != i:
                    self.electrode_potentials[i *
                                              (self.mesh.electrode_num - 1) + cnt] = np.abs(v)
                    cnt += 1
        return np.copy(self.electrode_potentials)