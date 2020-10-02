# @author: Li Zehao <https://philipli.art>
# @license: MIT
import numpy as np
import os
from .utilities import save_parameter
from .utilities import get_config
from .utilities import read_parameter
from .readmesh import read_mesh_from_csv


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
        self.mesh = read_mesh_from_csv()
        self.mesh_obj, _, _, _ = self.mesh.return_mesh()
        self.nodes = self.mesh_obj["node"]
        self.elem = self.mesh_obj["element"]
        self.point_x = self.nodes[:, 0]
        self.point_y = self.nodes[:, 1]
        self.read_JAC()
        self.elem_param = np.zeros((np.shape(self.elem)[0], 9))
        self.initialize()
        self.calc_detection_elements()
        if os.path.exists(self.config["folder_name"] + '/inv_mat.npy'):
            self.read_inv_matrix()
        else:
            self.read_JAC()
            self.get_inv_matrix()

    def return_mesh_info(self):
        return self.point_x, self.point_y, self.elem, self.detection_elem

    def initialize(self):
        """
        Update parameters for each element,

        Parameters used for calculating sparse matrix
        Calculate electrodes' mesh area
        initialize all electrode
        """
        x = [.0, .0, .0]
        b = [.0, .0, .0]
        c = [.0, .0, .0]
        y = [.0, .0, .0]
        count = 0
        for element in self.elem:

            # change to counter clockwise
            for i in range(3):
                x[i] = self.nodes[element[i], 0]
                y[i] = self.nodes[element[i], 1]
            parameter_mat = np.array([x, y, [1, 1, 1]])
            parameter_mat = parameter_mat.T  # Trans to vertical
            area = np.abs(np.linalg.det(parameter_mat) / 2)
            parameter_mat = np.linalg.inv(parameter_mat)  # get interpolation parameters
            parameter_mat = parameter_mat.T
            b = list(parameter_mat[:, 0])
            c = list(parameter_mat[:, 1])
            x_average = np.mean(x)  # get center point coordinate
            y_average = np.mean(y)
            self.elem_param[count] = [area, b[0], b[1], b[2], c[0], c[1], c[2], x_average, y_average]
            count += 1

    def calc_detection_elements(self):
        """
        Get elements whose center is inside detection range
        """
        original_element = self.elem
        original_x = self.elem_param[:, 7]
        original_y = self.elem_param[:, 8]
        corres_index = []
        new_elem = []
        for i, element in enumerate(original_element):
            if np.abs(original_x[i]) < self.config["detection_bound"] and np.abs(original_y[i]) < self.config["detection_bound"]:
                corres_index.append(i)
                new_elem.append(element)
        self.detection_index = np.array(corres_index)
        self.detection_elem = np.array(new_elem)

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
        for j in self.detection_index:
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
        Get the capacitance map according to delta_V Change

        Args:
            delta_V: voltage change on every dimensions the dimension should be (elec_num - 1) * elec_num

        Returns:
            capacitance density prediction inside detection area.
        """

        capacitance_predict = np.dot(self.inv_mat, delta_V)
        return capacitance_predict

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
        capacitance = self.solve(delta_V)
        idx = np.where(capacitance == np.max(capacitance))
        x = self.elem_param[idx][7]
        y = self.elem_param[idx][8]
        # Pick zone

    def delete_outside_detect(self, list_c):
        """
        Args:
            list_c : an all-element-wise list

        Returns:
             elements remained in detection domain
        """
        list_c = np.array(list_c)
        if list_c.ndim > 1:
            new_list_c = np.zeros((self.detection_index.shape[0], list_c.shape[1]))
            for i, j in enumerate(self.detection_index):
                new_list_c[i] = list_c[j]
            return new_list_c
        elif list_c.ndim == 1:
            new_list_c = np.zeros((self.detection_index.shape[0]))
            for i, j in enumerate(self.detection_index):
                new_list_c[i] = list_c[j]
            return new_list_c
        else:
            raise Exception("Transfer Shape Not Correct")


def reinitialize_solver():
    """
    Reinitialize EIT solver,

    Deletes the inv_mat.npy file and regenerate it.

    """
    config = get_config()
    if os.path.exists(config["folder_name"] + '/inv_mat.npy'):
        os.remove(config["folder_name"] + '/inv_mat.npy')
    new_solver = Solver()

