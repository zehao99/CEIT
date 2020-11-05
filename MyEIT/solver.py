# @author: Li Zehao <https://philipli.art>
# @license: MIT
import numpy as np
import os
from .util.utilities import get_config
from .readmesh import read_mesh_from_csv
from matplotlib import patches
from .models.mesh import MeshObj

class Solver(object):
    """
    Lite Jacobian Solver,

    You must first generate the jacobian matrix before using this.
    Use this in realtime reconstruciton.

    The parameters are generated in initialize process

    just call solve() for solving problem

    Methods:
        solve(): solver you should call
    """

    def __init__(self):
        print("Please make sure info in config.json is all correct, HERE WE GO!")
        self.config = get_config()
        mesh_obj, electrode_num, electrode_center_list, electrode_radius = read_mesh_from_csv().return_mesh()
        self.mesh = MeshObj(mesh_obj, electrode_num, electrode_center_list, electrode_radius)
        self.read_JAC()
        self.elem_param = np.zeros((np.shape(self.mesh.elem)[0], 9))
        if os.path.exists(self.config["folder_name"] + '/inv_mat.npy'):
            self.read_inv_matrix()
        else:
            self.read_JAC()
            self.get_inv_matrix()

    def return_mesh_info(self):
        """

        Returns:
            point_x: x axis of nodes
            point_y: y axis of nodes
            elem: elements with node num sets
            detection_elem: elements inside detection area
        """
        return self.mesh.point_x, self.mesh.point_y, self.mesh.elem, self.mesh.detection_elem

    def read_JAC(self):
        """
        Read JAC matrix from file
        """
        assert os.path.exists(self.config["folder_name"] + '/' + 'JAC_cache.npy'), "The JAC matrix is not generated"
        self.JAC_mat = np.load(self.config["folder_name"] + '/' + 'JAC_cache.npy')

    def eliminate_non_detect_JAC(self):
        """
        Delete all the elements outside detection range, the order should be identical to the element
        """
        orig_JAC = np.copy(self.JAC_mat.T)
        new_JAC = []
        for j in self.mesh.detection_index:
            new_JAC.append(orig_JAC[j, :])
        new_JAC = np.array(new_JAC)
        # save_parameter(new_JAC,'detect_JAC')
        return new_JAC.T

    def read_inv_matrix(self):
        """
        Read Inverse matrix cache
        """
        self.inv_mat = np.load(self.config["folder_name"] + '/inv_mat.npy')

    def solve(self, delta_V):
        """
        Realtime solver,
        Get the variable map according to delta_V Change

        Args:
            delta_V: voltage change on every dimensions the dimension should be (elec_num - 1) * elec_num

        Returns:
            variable density prediction inside detection area.
        """

        variable_predict = np.dot(self.inv_mat, delta_V)
        return variable_predict

    def get_inv_matrix(self, lmbda=203):
        """
        Calculate and save inverse matrix according to lmbda

        Args:
            lmbda: parameter for regularization

        """
        self.read_JAC()
        J = self.eliminate_non_detect_JAC() - 1
        Q = np.eye(J.shape[1])
        self.inv_mat = np.dot(np.linalg.inv(np.dot(J.T, J) + lmbda ** 2 * Q), J.T)
        np.save(self.config["folder_name"] + '/inv_mat.npy', self.inv_mat)

    def adaptive_solver(self, delta_V):
        """
        NOT COMPLETED
        """
        pass
    
    def plot_map_in_detection_range(self, ax, param):
        """
        Plot the current variable map,

        Args:
            ax: matplotlib.pyplot axis class
            param: parameter to be plotted(must match with the element)
        Returns:
            NULL
        """
        x, y = self.mesh.nodes[:, 0], self.mesh.nodes[:, 1]
        im = ax.tripcolor(x, y, self.mesh.detection_elem, np.abs(param), shading='flat')
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


def reinitialize_solver():
    """
    Reinitialize EIT solver,

    Deletes the inv_mat.npy file and regenerate it.

    """
    config = get_config()
    if os.path.exists(config["folder_name"] + '/inv_mat.npy'):
        os.remove(config["folder_name"] + '/inv_mat.npy')
    Solver()

