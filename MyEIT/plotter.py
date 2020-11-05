from MyEIT.util.utilities import get_config
from .readmesh import read_mesh_from_csv
from .models.mesh import MeshObj
import numpy as np
import matplotlib.patches as patches


class EITPlotter(object):

    def __init__(self):
        self.config = get_config()
        self.mesh = MeshObj()

    def plot_distribution(self, data, ax):

        x, y = self.mesh.nodes[:, 0], self.mesh.nodes[:, 1]
        im = ax.tripcolor(x, y, self.mesh.detection_elem, np.abs(data), shading='flat', cmap='plasma')
        ax.set_aspect('equal')
        radius = self.mesh.electrode_radius
        for i, electrode_center in enumerate(self.mesh.electrode_center_list):
            x0 = electrode_center[0] - radius
            y0 = electrode_center[1] - radius
            width = 2 * radius
            ax.add_patch(
                patches.Rectangle(
                    (x0, y0),  # (x,y)
                    width,  # width
                    width,  # height
                    color='k'
                )
            )
