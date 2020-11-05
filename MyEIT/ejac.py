
# @author: Li Zehao <https://philipli.art>
# @license: MIT
import csv

# import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import progressbar
from matplotlib import patches
from .efem import EFEM
from MyEIT.util.utilities import get_config

"""
Depend on efem.py
"""


class EJAC(object):
    """
    calculate Jacobian matrix for the problem

    DEFAULT PERMITTIVITY IS SET TO 1/10000 SIEMENS

    Functions:

                JAC_calculation(self): Calculate and return JAC Matrix, Auto Save to File

                eit_solve(self, detect_potential, lmbda): Solve inverse problems

                eit_solve_delta_V(self, detect_potential_diff,lmbda)

                read_JAC_np(self): Load JAC matrix from file

                save_JAC_np(self): Cache the JAC matrix into file

                save_inv_matrix(self, lmbda): Calculate the inverse matrix and save to file (Preparation for realtime calculation)

                eit_solve_direct(self, detect_potential): High speed reconstruction(Based on inverse matrix MUST RUN save_inv_matrix())

                show_JAC(self): Show the jacobian in plot

                plot_potential(self, filename = 'potential_Temp'): Plot potentials on every electrode

                plot_sensitivity(self, area_normalization = True): Plot sensitivity map (Sum of columns in JAC_Matrix)
    """

    def __init__(self, mesh):
        """
        Prepare for calculation,

        Args:
            mesh : DICT element mesh See readmesh.py

        """

        self.config = get_config()
        self.mode = 'n'
        self.first = self.config["is_first_JAC_calculation"]
        self.detection_bound = self.config["detection_bound"]
        self.overall_variable = self.config["overall_origin_variable"]
        # Create FEM class
        # Set overall permitivity as the conductive sheet
        permi = 1 / 10000
        self.electrode_centers = self.config["electrode_centers"]
        self.fwd_FEM = EFEM(mesh)
        self.electrode_num = len(self.electrode_centers)
        # Initialize Matrix
        self.pattern_num = self.electrode_num * (self.electrode_num - 1)
        self.elem_num = self.fwd_FEM.elem_num
        self.electrode_original_potential = np.zeros((self.pattern_num))
        # Choose detection_area
        self.calc_detection_elements()
        self.JAC_matrix = np.zeros((self.pattern_num, self.elem_num))
        if self.mode == 'n':
            self.fwd_FEM.reset_variable(overall_variable=self.overall_variable)  # Set overall initial value
        elif self.mode == 'i':
            self.fwd_FEM.reset_variable_to_initial(0)  # Set initial value
        else:
            raise Exception('No Such Mode, Please check.')
        self.initial_variable = np.copy(self.fwd_FEM.elem_variable)
        self.calc_origin_potential()

    def calc_origin_potential(self):
        """
        calculate origin potential vector
        """
        for i in range(self.electrode_num):
            count = 0
            _, _, electrode_potential = self.fwd_FEM.calculation(i)
            for m in range(self.electrode_num):
                if m != i:
                    self.electrode_original_potential[i * (self.electrode_num - 1) + count] = np.abs(
                        electrode_potential[m])
                    count += 1

    def JAC_calculation(self):
        """
        calculate JAC matrix

        Returns:
            JAC Matrix
        """
        calc_from = self.config["calc_from"]
        calc_end = self.config["calc_end"]
        variable_change = self.config["variable_change_for_JAC"]
        if self.first:
            if calc_from > 0:
                self.read_JAC_np()
            # Changed matrix
            print('This might take a whilllllllllle............')
            for i in range(self.electrode_num):
                if i < calc_from:
                    continue
                if i >= calc_end:
                    break
                self.save_JAC_np()
                print("iteration: " + str(
                    i) + "     If you want to break, press Ctrl + C. But REMEMBER THIS iteration num!")
                for j in progressbar.progressbar(range(self.elem_num)):
                    # variable change in JAC
                    self.fwd_FEM.elem_variable[j] += variable_change
                    _, _, electrode_potential = self.fwd_FEM.calculation(i)
                    count = 0
                    for m in range(self.electrode_num):
                        if m != i:
                            self.JAC_matrix[i * (self.electrode_num - 1) + count][j] = np.abs(electrode_potential[m])
                            count += 1
                    self.fwd_FEM.elem_variable[j] -= variable_change
            # Minus and broadcast original value calculate differiential value
            self.save_JAC_np()
            print('Congrats! You made it!')
            return self.JAC_matrix
        else:
            self.read_JAC_np()
            return self.JAC_matrix

    def calc_detection_elements(self):
        """
        Calculate elements inside the dectection area specified
        """
        original_element = self.fwd_FEM.elem
        original_x = self.fwd_FEM.elem_param[:, 7]
        original_y = self.fwd_FEM.elem_param[:, 8]
        corres_index = []
        new_elem = []
        for i, element in enumerate(original_element):
            x_val = 0
            y_val = 0
            for idx in element:
                x_val += self.fwd_FEM.nodes[idx][0]
                y_val += self.fwd_FEM.nodes[idx][1]
            x_val /= 3
            y_val /= 3
            if np.abs(x_val) < self.detection_bound and np.abs(y_val) < self.detection_bound:
                corres_index.append(i)
                new_elem.append(element)
        self.detection_index = np.array(corres_index)
        self.detection_elem = np.array(new_elem)

    def eliminate_non_detect_JAC(self):
        """
        Eliminate the rows inside JAC Matrix where element is not used in detection area
        """
        orig_JAC = np.copy(self.JAC_matrix.T)
        new_JAC = []
        for j in self.detection_index:
            new_JAC.append(orig_JAC[j, :])
        new_JAC = np.array(new_JAC)
        # save_parameter(new_JAC,'detect_JAC')
        return new_JAC.T

    def eit_solve(self, detect_potential, lmbda=295):
        """
        detect_potential: electrode_num * (electrode_num - 1) elements NDArray vector

        lmbda: FLOAT regularization parameter
        """
        # self.normalize_sensitivity()
        J = self.eliminate_non_detect_JAC() - 1
        Q = np.eye(J.shape[1])  # * area_list
        # Q = np.diag(np.dot(J.T,J))
        delta_V = detect_potential - np.copy(self.electrode_original_potential)
        variable_predict = np.dot(np.dot(np.linalg.inv(np.dot(J.T, J) + lmbda ** 2 * Q), J.T), delta_V)
        # variable_predict = self.Msolve_gpu(J, Q, lmbda, delta_V)
        # self.plot_potential(delta_V, orig_ratio = 0)
        return variable_predict

    def eit_solve_delta_V(self, delta_V, lmbda=295):
        """
        detect_potential: electrode_num * (electrode_num - 1) elements NDArray vector

        lmbda: FLOAT regularization parameter
        """
        J = self.eliminate_non_detect_JAC()
        Q = np.eye(J.shape[1])  # * area_list
        variable_predict = np.dot(np.dot(np.linalg.inv(np.dot(J.T, J) + lmbda ** 2 * Q), J.T), delta_V)
        # variable_predict = self.Msolve_gpu(J, Q, lmbda, delta_V)
        return variable_predict

    def eit_solve_4electrodes(self, detect_potential, lmbda=90):
        return self.eit_solve_on_some_electrodes([2, 6, 10, 14], detect_potential, lmbda, mode="direct")

    def eit_solve_4electrodes_delta_V(self, delta_V, lmbda=90):
        return self.eit_solve_on_some_electrodes([2, 6, 10, 14], delta_V, lmbda, mode="diff")

    def eit_solve_8electrodes(self, detect_potential, lmbda=265):
        return self.eit_solve_on_some_electrodes([0, 2, 4, 6, 8, 10, 12, 14], detect_potential, lmbda, mode="direct")

    def eit_solve_8electrodes_delta_V(self, delta_V, lmbda=265):
        return self.eit_solve_on_some_electrodes([0, 2, 4, 6, 8, 10, 12, 14], delta_V, lmbda, mode="diff")

    def eit_solve_on_some_electrodes(self, electrode_list, potential, lmbda, mode="diff"):
        """
            Template for solving problem on some electrodes

            Args:
                electrode_list: electrode chosen
                potential: potential data
                lmbda: regularization parameters
                mode: solving the problem with difference of amplitude or amplitude raw data.

            Returns:

        """
        assert mode == "diff" or mode == "direct", "Please enter the right solver mode"
        slice_list = []
        for i in electrode_list:
            for j in electrode_list:
                if j == i:
                    pass
                if j < i:
                    slice_list.append(i * (self.electrode_num - 1) + j)
                if j > i:
                    slice_list.append(i * (self.electrode_num - 1) + j - 1)
        J = self.eliminate_non_detect_JAC() - 1
        J = J[slice_list, :]
        Q = np.eye(J.shape[1])
        if mode == "direct":
            delta_V = potential - np.copy(self.electrode_original_potential)
            delta_V = np.copy(delta_V[slice_list])
        else:
            delta_V = potential
        variable_predict = np.dot(np.dot(np.linalg.inv(np.dot(J.T, J) + lmbda ** 2 * Q), J.T), delta_V) * 1e-4
        return variable_predict

    def save_inv_matrix(self, lmbda=203):
        """
        Calculate and save inverse matrix according to lmbda
        (Preparation for realtime calculation)

        Args:
            lmbda: regularization parameter
        """
        J = self.eliminate_non_detect_JAC() - 1
        Q = np.eye(J.shape[1])
        JAC_inv = np.dot(np.linalg.inv(np.dot(J.T, J) + lmbda ** 2 * Q), J.T)
        np.save(self.config["folder_name"] + '/inv_mat.npy', JAC_inv)

    def read_inv_matrix(self):
        """
        Load inverse matrix from inv_mat.npy
        """

        return np.load(self.config["folder_name"] + '/inv_mat.npy')

    def eit_solve_direct(self, detect_potential):
        """
        Use saved inverse matrix to solve
        """
        JAC_p = self.read_inv_matrix()
        delta_V = detect_potential - np.copy(self.electrode_original_potential)
        variable_predict = np.dot(JAC_p, delta_V)
        return variable_predict

    def save_JAC_np(self):
        """
        save JAC matrix to JAC_cache.npy
        """
        np.save(self.config["folder_name"] + '/' + 'JAC_cache.npy', self.JAC_matrix)

    def read_JAC_np(self):
        """
        read JAC matrix to JAC_cache.npy
        """
        self.JAC_matrix = np.load(self.config["folder_name"] + '/' + 'JAC_cache.npy')

    def save_JAC_2file(self):
        """
        Save jacobian matrix to csv file
        """
        with open(self.config["folder_name"] + '/' + 'jac_cache.csv', "w", newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for row in self.JAC_matrix:
                writer.writerow(row)

    def read_JAC_2file(self, mode='normal'):
        """
        Read jacobian matrix from csv file
        """
        if mode == 'normal':
            with open(self.config["folder_name"] + '/' + 'jac_cache.csv', newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for i, line in enumerate(reader):
                    self.JAC_matrix[i] = line
        elif mode == 'sudoku':
            self.JAC_matrix = np.zeros((self.pattern_num, 9))
            with open(self.config["folder_name"] + '/' + 'jac_cache.csv', newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for i, line in enumerate(reader):
                    self.JAC_matrix[i] = line
        else:
            raise Exception('Mode do not exist.')

    def show_JAC(self):
        """
        Show JAC matrix
        """
        plt.imshow(self.JAC_matrix[:, :])
        plt.show()

    def get_sensitivity_list(self):
        """
        Sum up all numbers under different pattern for a single element in JAC_matrix
        """
        sensitivity = np.sum(self.JAC_matrix, axis=0)
        return sensitivity

    def normalize_sensitivity(self):
        """
        DEPRECATED
        Normalize JAC by sensitivity
        Didn't work.....
        """
        sensitivity = self.get_sensitivity_list()
        self.JAC_matrix = self.JAC_matrix / sensitivity.T * np.mean(sensitivity)

    def delete_outside_detect(self, list_c):
        """
        Input a all-element-wise list
        Return elements remained in detection domain
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

    def plot_sensitivity(self, area_normalization=True):
        """
        Plot sensitivity map
        """
        if area_normalization:
            sensitivity = self.get_sensitivity_list() / self.fwd_FEM.elem_param[:, 0]
        else:
            sensitivity = self.get_sensitivity_list()
        points = self.fwd_FEM.nodes
        tri = self.fwd_FEM.elem
        x, y = points[:, 0], points[:, 1]
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.tripcolor(x, y, tri, sensitivity, shading='flat')
        fig.colorbar(im)
        ax.set_aspect('equal')
        plt.show()
    
    def plot_map_in_detection_range(self, ax, param):
        """
        Plot the current variable map,

        Args:
            ax: matplotlib.pyplot axis class
            param: parameter to be plotted(must match with the element)
        Returns:
            NULL
        """
        x, y = self.fwd_FEM.nodes[:, 0], self.fwd_FEM.nodes[:, 1]
        im = ax.tripcolor(x, y, self.detection_elem, np.abs(param), shading='flat')
        ax.set_aspect('equal')
        radius = self.fwd_FEM.electrode_radius
        for i, electrode_center in enumerate(self.fwd_FEM.electrode_center_list):
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

    # def Msolve_gpu(self, J, Q, lmbda, delta_V):
    #     """
    #     GPU Acceleration
    #     """
    #     J_g = cp.asarray(J)
    #     J_g_T = cp.asarray(J.T)
    #     Q_g = cp.asarray(Q)
    #     delta_V_g = cp.asarray(delta_V)
    #     variable_predict = cp.dot(cp.dot(cp.linalg.inv(cp.dot(J_g_T, J_g) + lmbda ** 2 * Q_g), J_g_T), delta_V_g)
    #     variable_predict = cp.asnumpy(variable_predict)
    #     return variable_predict
