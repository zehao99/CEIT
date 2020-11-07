from .util.utilities import get_config
from .readmesh import read_mesh_from_csv
from .models.mesh import MeshObj
import numpy as np
import matplotlib.patches as patches


class EITPlotter(object):
    """
    Plotter for plotting distribution of variables inside the problem.
    """
    def __init__(self, mesh=None):
        """
        Initialize the plotter

        If an mesh object is already created, please pass it into the plotter to save memory use.

        Args:
            mesh: ``MeshObj`` object
        """
        self.config = get_config()
        if mesh is None:
            self.mesh = MeshObj()
        else:
            self.mesh = mesh

    def plot_detection_area_map(self, param, ax, with_electrode=False):
        """
            Plot the current variable map inside detection area.

            The detection area is the area inside ``detection_bound`` and excluding the electrode meshes.

            TURNING with_electrode ON WILL AFFECT PERFORMANCE

            Args:
                param: parameter to be plotted(must match with the element)
                ax: matplotlib.pyplot axis class
                with_electrode: whether to add electrodes inside plot
            Returns:
                im: matplotlib image
        """
        im = self.plot_with_electrode_shown_helper(
            self.mesh.detection_elem, np.abs(param), ax, with_electrode, cmap='viridis')
        return im

    def plot_full_area_map(self, param, ax, with_electrode=False):
        """
        Plot the current variable map inside whole mesh area.

        TURNING with_electrode ON WILL AFFECT PERFORMANCE

        Args:
            param: parameter to be plotted(must match with the element)
            ax: matplotlib.pyplot axis class
            with_electrode: whether to add electrodes inside plot
        Returns:
            im: matplotlib image object
        """
        im = self.plot_with_electrode_shown_helper(
            self.mesh.elem, np.abs(param), ax, with_electrode)
        return im

    def plot_with_electrode_shown_helper(self, elements, param, ax, with_electrode=False, cmap='plasma'):
        """
        Plot the current variable map,

        Args:
            elements: elements of the base mesh
            ax: matplotlib.pyplot axis class
            param: parameter to be plotted(must match with the element)
            with_electrode: whether to add electrodes inside plot
            cmap: Color map options for ``ax.tripcolor``
        Returns:

        """
        x, y = self.mesh.nodes[:, 0], self.mesh.nodes[:, 1]
        im = ax.tripcolor(x, y, elements, param, shading='flat', cmap=cmap)
        ax.set_aspect('equal')
        if with_electrode:
            self.add_electrodes_to_axes(ax)
        return im

    def add_electrodes_to_axes(self, ax):
        """
        Add electrodes to the axes.

        Args:
            ax: matplotlib.axes object.
        """
        radius = self.mesh.electrode_radius
        perimeter = self.mesh.get_perimeter()
        p_x = []
        p_y = []
        for idx in perimeter:
            p_x.append(self.mesh.nodes[idx][0])
            p_y.append(self.mesh.nodes[idx][1])
        p_x.append(self.mesh.nodes[perimeter[0]][0])
        p_y.append(self.mesh.nodes[perimeter[0]][1])
        ax.plot(p_x, p_y, color='k')
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