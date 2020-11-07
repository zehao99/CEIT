# @author: Li Zehao <https://philipli.art>
# @license: MIT
import math
# import cupy as cp
import numpy as np
from .FEMBasic import FEMBasic


class EFEM(FEMBasic):
    """
    FEM solving for variable included forward problem,

    Functions:
                calculation(self,electrode_input): Forward Calculation

                plot_potential_map: Plot the forward result

                plot_current_variable: Plot the variable ground truth

                change_variable_elementwise(self, element_list, variable_list): Change variable

                change_variable_geometry(self, center, radius, value, shape): Change variable

                change_add_variable_geometry(self, center, radius, value, shape): Adding variable to region

                change_conductivity(self, element_list, resistance_list): Change conductivity

                reset_variable(self, overall_variable):Reset all variable to 0

                reset_variable_to_initial(self, variable_value): DEPRECATED Reset with radius 20 square value 10e-8.46

    """

    def __init__(self, mesh=None):
        super().__init__(mesh)

    def my_solver(self, electrode_input):
        """
        My version of solver
        """
        theta = np.float(0)
        self.construct_sparse_matrix()  # 0.1343s
        self.set_boundary_condition(electrode_input)  # 0.005s
        self.node_potential = np.abs(self.calculate_FEM(theta))  # 0.211s
        self.sync_back_potential()  # 0.001s

    def construct_sparse_matrix(self):
        """
        construct the original sparse matrix 
        """
        index = 0
        K_ij = 0 + 0j
        pattern = [[0, 0], [1, 1], [2, 2], [0, 1], [1, 2], [2, 0], [1, 0], [2, 1],
                   [0, 2]]  # go through every combination of k1 and k2
        for element in self.mesh.elem:
            param = self.mesh.elem_param[index]
            for i, j in pattern:
                if i != j:
                    # stiffness k_ij = sigma * (bk1*bk2 + ck1*ck2)/(4 * area) - j * w * variable * (bk1 * ck2 -
                    # bk2 * ck1) /24
                    K_ij = self.mesh.elem_perm[index] * (param[1 + i] * param[1 + j] + param[4 + i] * param[4 + j]) * (
                        1 * param[0]) - (self.freq * self.elem_variable[index] * param[0] / 12) * 1j
                    self.K_sparse[element[i]][element[j]] += K_ij
                    self.K_sparse[element[j]][element[i]] += K_ij
                else:
                    K_ij = self.mesh.elem_perm[index] * (param[1 + i] * param[1 + j] + param[4 + i] * param[4 + j]) * (
                        1 * param[0]) - (self.freq * self.elem_variable[index] * param[0] / 6) * 1j
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
        electrode_list = list(self.mesh.electrode_mesh.values())
        for element in electrode_list[electrode_input]:
            node_list.append(self.mesh.elem[element][0])
            node_list.append(self.mesh.elem[element][1])
            node_list.append(self.mesh.elem[element][2])
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
        Solve forward problem
        """
        # changing theta could help increasing the accuracy
        # set the phi_f and phi_b
        potential_f = np.zeros((self.node_num_f, 1), dtype=np.complex128)
        potential_b = (np.cos(theta) + 1j * math.sin(theta)) * \
            np.ones((self.node_num_bound, 1))
        if self.node_num_ground != 0:
            potential_g = np.zeros(
                (self.node_num_ground, 1), dtype=np.complex128)
            potential_b = np.append(potential_g, potential_b)
        K_f = self.K_sparse[0: self.node_num_f, 0: self.node_num_f]
        K_b = self.K_sparse[0: self.node_num_f, self.node_num_f: self.node_num]
        # solving the linear equation set
        if self.config["device"] == "gpu":
            potential_f = calculate_FEM_equation(
                potential_f, K_f, K_b, potential_b)  # GPU_Method faster
        elif self.config["device"] == "cpu":
            potential_f = - \
                np.dot(np.dot(np.linalg.inv(K_f), K_b), potential_b)
        else:
            raise Exception(
                'Please make sure you specified device inside the config file to \"cpu\" or \"gpu\"')
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
    pass
    # K_f_gpu = cp.asarray(K_f)
    # K_b_gpu = cp.asarray(K_b)
    # potential_b_gpu = cp.asarray(potential_b)
    # result_gpu = - cp.dot(cp.dot(cp.linalg.inv(K_f_gpu),
    #                              K_b_gpu), potential_b_gpu)
    # result = cp.asnumpy(result_gpu)
    # return result  # solving the linear equation set
