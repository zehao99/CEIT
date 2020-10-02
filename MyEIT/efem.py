import math
from .utilities import get_config
import cupy as cp
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from .fem import FEM_Basic


class EFEM(FEM_Basic):
    """
    FEM solving for capacitance included forward problem,

    Functions:
                calculation(self,electrode_input): Forward Calculation

                plot_potential_map: Plot the forward result

                plot_current_capacitance: Plot the capacitance ground truth

                change_capacitance_elementwise(self, element_list, capacitance_list): Change capacitance

                change_capacitance_geometry(self, center, radius, value, shape): Change capacitance
                
                change_add_capa_geometry(self, center, radius, value, shape): Adding capacitance to region

                change_conductivity(self, element_list, resistance_list): Change conductivity

                reset_capacitance(self, overall_capa):Reset all capacitance to 0
                
                reset_capa_to_initial(self, capacitance_value): DEPRECATED Reset with radius 20 square value 10e-8.46

    """

    def __init__(self, mesh):
        super().__init__(mesh)

    def plot_capacitance_map(self, ax):
        """
        Plot the current capacitance map,

        Args:
            ax: matplotlib.pyplot axis class
        Returns:
            Image
        """
        im = self.plot_map(ax, self.elem_capacitance)
        return im

    def plot_potential_map(self, ax):
        """
            Plot the current capacitance map,

            Args:
                ax: matplotlib.pyplot axis class
            Returns:
                Image
        """
        im = self.plot_map(ax, self.element_potential)
        return im

    def change_capacitance_elementwise(self, element_list, capacitance_list):
        """Change capacitance in certain area according to ELEMENT NUMBER,

        Args:
            element_list: INT LIST element numbers to be changed
            capacitance_list: FLOAT LIST same dimension list for capacitance on each element included
        Returns:
            NULL
        """
        if len(element_list) == len(capacitance_list):
            for i, ele_num in enumerate(element_list):
                if ele_num > self.elem_num:
                    raise Exception("Element number exceeds limit")
                self.elem_capacitance[ele_num] = capacitance_list[i]
        else:
            raise Exception('The length of element doesn\'t match the length of capacitance')

    def change_capacitance_geometry(self, center, radius, value, shape):
        """Change capacitance in certain area according to GEOMETRY,

        Args:
            center: [FLOAT , FLOAT] center of the shape
            radius: FLOAT radius (half side length) of the shape
            value: FLOAT area capacitance value
            shape: STR "circle", "square"
        Returns:
            NULL
        Raises:
            No element is selected, please check the input
            No such shape, please check the input
        """
        if shape == "square":
            center_x, center_y = center
            count = 0
            for i, x in enumerate(self.elem_param[:, 7]):
                if (center_x + radius) >= x >= (center_x - radius) and (
                        center_y + radius) >= self.elem_param[i][8] >= (center_y - radius):
                    self.elem_capacitance[i] = value
                    count += 1
            if count == 0:
                raise Exception("No element is selected, please check the input")
        elif shape == "circle":
            center_x, center_y = center
            count = 0
            for i, x in enumerate(self.elem_param[:, 7]):
                if np.sqrt((center_x - x) ** 2 + (center_y - self.elem_param[i][8]) ** 2) <= radius:
                    self.elem_capacitance[i] = value
                    count += 1
        else:
            raise Exception("No such shape, please check the input")

    def change_add_capa_geometry(self, center, radius, value, shape):
        """Add capacitance in certain area according to GEOMETRY

        Args:
            center: [FLOAT , FLOAT] center of the shape
            radius: FLOAT radius (half side length) of the shape
            value: FLOAT area capacitance value
            shape: STR "circle", "square"
        Returns:
            NULL
        Raises:
            No element is selected, please check the input
            No such shape, please check the input
        """
        if shape == "square":
            center_x, center_y = center
            count = 0
            for i, x in enumerate(self.elem_param[:, 7]):
                if (center_x + radius) >= x >= (center_x - radius) and (
                        center_y + radius) >= self.elem_param[i][8] >= (center_y - radius):
                    self.elem_capacitance[i] += value
                    count += 1
            if count == 0:
                raise Exception("No element is selected, please check the input")
        elif shape == "circle":
            center_x, center_y = center
            count = 0
            for i, x in enumerate(self.elem_param[:, 7]):
                if np.sqrt((center_x - x) ** 2 + (center_y - self.elem_param[i][8]) ** 2) <= radius:
                    self.elem_capacitance[i] += value
                    count += 1
        else:
            raise Exception("No such shape, please check the input")

    def reset_capacitance(self, overall_capa=0):
        """
        Set capacitance on every value to overall_capa,

        Args:
            overall_capa: FLOAT target capacitance value for every element
        """
        self.elem_capacitance = np.zeros(np.shape(self.elem_perm)) + overall_capa

    def reset_capa_to_initial(self, capacitance_value):
        """
        DEPRECATED
        Set initial distribution of capacitance density value
        """
        self.elem_capacitance = np.zeros(np.shape(self.elem_perm))
        self.change_capacitance_geometry([0, 0], 15, capacitance_value, shape="square")

    def change_conductivity(self, element_list, resistance_list):
        """
        Change conductivity in certain area according to ELEMENT NUMBER

         Args:
            element_list: INT LIST element numbers to be changed
            resistance_list: FLOAT LIST same dimension list for conductivity on each element included
        """
        if len(element_list) == len(resistance_list):
            for i, ele_num in enumerate(element_list):
                if ele_num > self.elem_num:
                    raise Exception("Element number exceeds limit")
                self.elem_perm[ele_num] = resistance_list[i]
        else:
            raise Exception('The length of element doesn\'t match the length of capacitance')

    def construct_sparse_matrix(self):
        """
        construct the original sparse matrix 
        """
        index = 0
        K_ij = 0 + 0j
        pattern = [[0, 0], [1, 1], [2, 2], [0, 1], [1, 2], [2, 0], [1, 0], [2, 1],
                   [0, 2]]  # go through every combination of k1 and k2
        for element in self.elem:
            param = self.elem_param[index]
            for i, j in pattern:
                if i != j:
                    # stiffness k_ij = sigma * (bk1*bk2 + ck1*ck2)/(4 * area) - j * w * capacitance * (bk1 * ck2 -
                    # bk2 * ck1) /24
                    K_ij = self.elem_perm[index] * (param[1 + i] * param[1 + j] + param[4 + i] * param[4 + j]) * (
                            1 * param[0]) - (self.freq * self.elem_capacitance[index] * param[0] / 12) * 1j
                    self.K_sparse[element[i]][element[j]] += K_ij
                    self.K_sparse[element[j]][element[i]] += K_ij
                else:
                    K_ij = self.elem_perm[index] * (param[1 + i] * param[1 + j] + param[4 + i] * param[4 + j]) * (
                            1 * param[0]) - (self.freq * self.elem_capacitance[index] * param[0] / 6) * 1j
                    self.K_sparse[element[i]][element[j]] += K_ij
                    self.K_sparse[element[j]][element[i]] += K_ij

            index += 1

    def set_boundary_condition(self, electrode_input):
        """
        Update boundary condition according to electrode mesh

        The boundary is at the input electrode whose potential is all Ae ^ iPhi

        And swap the index of matrix put the boundary elements at the bottom of the sparse matrix
        """
        node_list = []  # reshape all nodes to 1D
        electrode_list = list(self.electrode_mesh.values())
        for element in electrode_list[electrode_input]:
            node_list.append(self.elem[element][0])
            node_list.append(self.elem[element][1])
            node_list.append(self.elem[element][2])
        node_list = np.array(node_list)
        node_list = list(np.unique(node_list))  # get rid of repeat numbers
        index = self.node_num
        # Swap the boundary condition to the end of vector
        self.node_num_bound = len(node_list)
        self.node_num_f = self.node_num - self.node_num_bound
        for list_num in node_list:
            if list_num < self.node_num_f:
                index = index - 1
                while index in node_list:
                    index = index - 1
                self.swap(list_num, index)

    def calculate_FEM(self, theta):
        """
        Solve forward problem,
        """
        # changing theta could help increasing the accuracy
        potential_f = np.zeros((self.node_num_f, 1), dtype=np.complex128)  # set the phi_f and phi_b
        potential_b = (np.cos(theta) + 1j * math.sin(theta)) * np.ones((self.node_num_bound, 1))
        if self.node_num_ground != 0:
            potential_g = np.zeros((self.node_num_ground, 1), dtype=np.complex128)
            potential_b = np.append(potential_g, potential_b)
        K_f = self.K_sparse[0: self.node_num_f, 0: self.node_num_f]
        K_b = self.K_sparse[0: self.node_num_f, self.node_num_f: self.node_num]
        # solving the linear equation set
        if self.config["device"] == "gpu":
            potential_f = calculate_FEM_equation(potential_f, K_f, K_b, potential_b)  # GPU_Method faster
        elif self.config["device"] == "cpu":
            potential_f = - np.dot(np.dot(np.linalg.inv(K_f), K_b), potential_b)
        else:
            raise Exception('Please make sure you specified device inside the config file to \"cpu\" or \"gpu\"')
        potential_f = np.reshape(potential_f, (-1))
        potential_b = np.reshape(potential_b, (-1))
        potential_f = np.append(potential_f, potential_b)
        return potential_f

    def sync_back_potential(self):
        """
        Put the potential back in order,
        """
        potential = np.copy(self.node_potential)
        for i, j in enumerate(self.K_node_num_list):
            self.node_potential[j] = potential[i]

    def calc_electrode_potential(self):
        """
        Get the mean value of potential on every electrode,
        """
        for i, elements in enumerate(self.electrode_mesh.values()):
            potential = []
            for element in elements:
                potential.append(self.element_potential[element])
            self.electrode_potential[i] = np.mean(np.array(potential))

    def swap(self, a, b):
        """
        Swap two rows and columns of the sparse matrix,
        """
        self.K_sparse[[a, b], :] = self.K_sparse[[b, a], :]
        self.K_sparse[:, [a, b]] = self.K_sparse[:, [b, a]]
        self.K_node_num_list[a], self.K_node_num_list[b] = self.K_node_num_list[b], self.K_node_num_list[a]


# Comment this function out if use cpu
def calculate_FEM_equation(potential_f, K_f, K_b, potential_b):
    """
    GPU acceleration for inverse calculation,
    """
    K_f_gpu = cp.asarray(K_f)
    K_b_gpu = cp.asarray(K_b)
    potential_b_gpu = cp.asarray(potential_b)
    result_gpu = - cp.dot(cp.dot(cp.linalg.inv(K_f_gpu), K_b_gpu), potential_b_gpu)
    result = cp.asnumpy(result_gpu)
    return result  # solving the linear equation set
