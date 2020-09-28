import math

import cupy as cp
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class EFEM(object):
    """
    FEM solving for capacitance included forward problem,

    Parameters:
                mesh: dict with "node", "element" "perm"
                        "node": N_node * 2 NDArray 
                            contains coordinate information of each node
                        "element": N_elem * 3 NDArray
                            contains the node number contained in each element
                        "perm": N_elem NDArray
                            contains permittivity on each element Default is 1
    
                electrode_nums:  INT electrode numbers
                                 PLEASE CALL calc_electrode_elements(...) FUNCTION BEFORE PROCEEDING
    
                electrode_center_list : electrode_nums * 2 NDArray
                                        electrode center info

                electrode_radius : FLOAT electrode radius(square shape) (a/2)

                frequency : FLOAT Input frequency

                perm : FLOAT Input overall permittivity


    Functions:
                calculation(self,electrode_input): Forward Calculation

                plot_potential_map: Plot the forward result

                plot_current_capacitance: Plot the capacitance ground truth

                change_capacitance_elementwise(self, element_list, capacitance_list): Change capacitance

                change_capacitance_geometry(self, center, radius, value, shape): Change capacitance
                
                change_add_capa_geometry(self, center, radius, value, shape): Adding capacitance to region

                change_conductivity(self, element_list, resistance_list): Change conductivity

                calc_electrode_elements(self, electrode_number, center, radius): Set electrode elements
                
                reset_capacitance(self, overall_capa):Reset all capacitance to 0
                
                reset_capa_to_initial(self, capacitance_value): Reset with radius 20 square value 10e-8.46

    """

    def __init__(self, mesh, electrode_nums, electrode_center_list, electrode_radius, frequency=20000.0 * 2 * np.pi,
                 perm=1):
        # check data structure
        self.nodes = mesh['node']
        self.node_num = np.shape(self.nodes)[0]
        if np.shape(self.nodes)[1] != 2:
            raise Exception("2D Node coordinates incorrect")
        self.node_num_bound = 0
        self.node_num_f = 0
        self.node_num_ground = 0
        self.elem = mesh['element']
        if np.shape(self.elem)[1] != 3:
            raise Exception("2D Elements dimension incorrect")
        self.elem_num = np.shape(self.elem)[0]
        self.elem_perm = mesh['perm'] * perm
        self.elem_capacitance = np.zeros(np.shape(self.elem_perm))
        self.elem_param = np.zeros((np.shape(self.elem)[0], 9))  # area, b1, b2, b3, c1, c2, c3, x_average, y_average
        self.electrode_num = electrode_nums
        self.electrode_center_list = electrode_center_list
        self.electrode_radius = electrode_radius
        self.electrode_mesh = dict()
        for i in range(electrode_nums):
            self.electrode_mesh[i] = list()
        self.freq = frequency
        self.K_sparse = np.zeros((self.node_num, self.node_num), dtype=np.complex128)
        self.K_node_num_list = [x for x in range(self.node_num)]  # Node number mapping list when calculating
        self.node_potential = np.zeros((self.node_num), dtype=np.complex128)
        self.element_potential = np.zeros((self.elem_num), dtype=np.complex128)
        self.electrode_potential = np.zeros((self.electrode_num), dtype=np.complex128)
        self.initialize_inv()

    def calculation(self, electrode_input):
        """
        Forward Calculation,

        Args:
            electrode_input: INT input position
        Returns:
            node_potential, element_potential, electrode_potential
        """
        self.calc_init()  # 0.053 s
        self.construct_sparse_matrix()  # 0.1343s
        self.set_boundary_condition(electrode_input)  # 0.005s
        self.set_boundary_condition_grounded()
        # split time into frames
        # frames = 1
        # temp_node_potential = np.zeros((self.node_num), dtype = np.complex128)
        # for i in range(frames):
        theta = np.float(0)  # Assign a phase for input
        # temp_node_potential += np.abs(self.calculate_FEM(theta))
        # temp_node_potential /= frames # 0.211s per step
        self.node_potential = np.abs(self.calculate_FEM(theta))  # 0.211s
        self.sync_back_potential()  # 0.001s
        self.calculate_element_potential()  # 0.004s
        self.calc_electrode_potential()  # 0.001s
        return self.node_potential, self.element_potential, self.electrode_potential

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

    def calc_init(self):
        """
        Set all parameters to zero,

        """
        self.K_sparse = np.zeros((self.node_num, self.node_num), dtype=np.complex128)
        self.K_node_num_list = [x for x in range(self.node_num)]  # Node number mapping list when calculating
        self.node_potential = np.zeros((self.node_num), dtype=np.complex128)
        self.element_potential = np.zeros((self.elem_num), dtype=np.complex128)
        self.electrode_potential = np.zeros((self.electrode_num), dtype=np.complex128)

    def initialize_inv(self):
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
            x_average = np.mean(x)  # get center point cordinate
            y_average = np.mean(y)
            self.elem_param[count] = [area, b[0], b[1], b[2], c[0], c[1], c[2], x_average, y_average]
            count += 1
        # Set electrode meshes(Square)
        for i in range(self.electrode_num):
            center = self.electrode_center_list[i]
            self.calc_electrode_elements(i, center, self.electrode_radius)

    def initialize(self):
        """
        DEPRECATED

        Update parameters for each element,

        Parameters used for calculating sparse matrix
        a1 = x2 * y3 - x3 * y2
        b1 = y2 - y3
        c1 = x3 - x2
        area = (b1 * c2 - b2 * c1) / 2
        Calculate electrodes' mesh area
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
            if ((y[1] - y[2]) * (x[0] - x[2]) - (y[2] - y[0]) * (x[2] - x[1])) / 2 < 0:
                self.elem[count][1], self.elem[count][2] = self.elem[count][2], self.elem[count][1]
                x[1], x[2] = x[2], x[1]
                y[1], y[2] = y[2], y[1]
            for i in range(3):
                b[i] = y[(1 + i) % 3] - y[(2 + i) % 3]
                c[i] = x[(2 + i) % 3] - x[(1 + i) % 3]
            area = (b[0] * c[1] - b[1] * c[0]) / 2
            x_average = np.mean(x)
            y_average = np.mean(y)
            self.elem_param[count] = [area, b[0], b[1], b[2], c[0], c[1], c[2], x_average, y_average]
            count += 1
        # Set electrode meshes(Square)
        for i in range(self.electrode_num):
            center = self.electrode_center_list[i]
            self.calc_electrode_elements(i, center, self.electrode_radius)

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

    def set_boundary_condition_grounded(self):
        ground_node_list = []
        index = self.node_num - self.node_num_bound
        for i, j in enumerate(self.elem_capacitance):
            if j >= 1e-3:
                ground_node_list.append(self.elem[i][0])
                ground_node_list.append(self.elem[i][1])
                ground_node_list.append(self.elem[i][2])
        ground_node_list = np.array(ground_node_list)
        ground_node_list = list(np.unique(ground_node_list))
        self.node_num_ground = len(ground_node_list)
        self.node_num_f = self.node_num_f - self.node_num_ground
        if self.node_num_ground != 0:
            for list_num in ground_node_list:
                if list_num < self.node_num_f:
                    index = index - 1
                    while index in ground_node_list:
                        index = index - 1
                    self.swap(list_num, index)

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
        potential_f = calculate_FEM_equation(potential_f, K_f, K_b, potential_b)  # GPU_Method faster
        # potential_f = - np.dot(np.dot(np.linalg.inv(K_f) , K_b) , potential_b) #solving the linear equation set
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

    def calculate_element_potential(self):
        """
        Get each element's potential,

        Average of each nodes on element
        """
        for i, _ in enumerate(self.elem_param):
            k1, k2, k3 = self.elem[i]
            self.element_potential[i] = (self.node_potential[k1] + self.node_potential[k2] + self.node_potential[
                k3]) / 3

    def calc_electrode_elements(self, electrode_number, center, radius):
        """
        Get the electrode element sets for every electrode,

        According to the SQUARE area given and put values into electrode_mesh dict

        Parameters: 
                    electrode_number: INT current electrode number
                    center: [FLOAT,FLOAT] center of electrode
                    radius: FLOAT half side length of electrode
        """
        if electrode_number >= self.electrode_num:
            raise Exception("the input number exceeded electrode numbers")
        else:
            center_x, center_y = center
            count = 0
            for i, x in enumerate(self.elem_param[:, 7]):
                if (center_x + radius) >= x >= (center_x - radius) and (
                        center_y + radius) >= self.elem_param[i][8] >= (center_y - radius):
                    self.electrode_mesh[electrode_number].append(i)
                    count += 1
            if count == 0:
                raise Exception("No element is selected, please check the input")

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

    def plot_map(self, ax, param):
        """
        Plot the current capacitance map,

        Args:
            ax: matplotlib.pyplot axis class
            param: parameter to be plotted(must match with the element)
        Returns:
            NULL
        """
        x, y = self.nodes[:, 0], self.nodes[:, 1]
        im = ax.tripcolor(x, y, self.elem, np.abs(param), shading='flat', cmap='plasma')
        ax.set_aspect('equal')
        radius = self.electrode_radius
        for i, electrode_center in enumerate(self.electrode_center_list):
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
            ax.add_patch(
                patches.Rectangle(
                    (-50, -50),  # (x,y)
                    100,  # width
                    100,  # height
                    color='k',
                    fill=False
                )
            )

        return im


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
