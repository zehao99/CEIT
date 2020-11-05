from .util.utilities import get_config
from .readmesh import read_mesh_from_csv
from .models.mesh import MeshObj
import numpy as np
import matplotlib.patches as patches


class EITPlotter(object):

    def __init__(self, mesh=None):
        self.config = get_config()
        if mesh is None:
            self.mesh = MeshObj()
        else:
            self.mesh = mesh

    def plot_detection_area_map(self, data, ax):
        """
            Plot the current variable map,

            Args:
                param: parameter to be plotted(must match with the element)
                ax: matplotlib.pyplot axis class
            Returns:
                im: matplotlib image
        """
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
        return im

    def plot_full_area_map(self, param, ax):
        """
        Plot the current variable map,

        Args:
            ax: matplotlib.pyplot axis class
            param: parameter to be plotted(must match with the element)
        Returns:
            NULL
        """
        x, y = self.mesh.nodes[:, 0], self.mesh.nodes[:, 1]
        im = ax.tripcolor(x, y, self.mesh.elem, np.abs(param), shading='flat', cmap='plasma')
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
        return im