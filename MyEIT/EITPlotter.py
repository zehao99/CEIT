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

    def plot_detection_area_map(self, param, ax):
        """
            Plot the current variable map,

            Args:
                param: parameter to be plotted(must match with the element)
                ax: matplotlib.pyplot axis class
            Returns:
                im: matplotlib image
        """
        im = self.plot_with_electrode_shown_helper(
            self.mesh.detection_elem, np.abs(param), ax, cmap='viridis')
        return im

    def plot_full_area_map(self, param, ax):
        """
        Plot the current variable map,

        Args:
            param: parameter to be plotted(must match with the element)
            ax: matplotlib.pyplot axis class
        Returns:
            NULL
        """
        im = self.plot_with_electrode_shown_helper(
            self.mesh.elem, np.abs(param), ax)
        return im

    def plot_with_electrode_shown_helper(self, elements, param, ax, cmap='plasma'):
        """
        Plot the current variable map,

        Args:
            elements: elements of the base mesh
            ax: matplotlib.pyplot axis class
            param: parameter to be plotted(must match with the element)
        Returns:
            NULL
        """
        x, y = self.mesh.nodes[:, 0], self.mesh.nodes[:, 1]
        im = ax.tripcolor(x, y, elements, param, shading='flat', cmap=cmap)
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
