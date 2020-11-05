from .utilities import get_config
import matplotlib.patches as patches
import numpy as np
from abc import ABCMeta, abstractmethod


class FEMBasic(metaclass=ABCMeta):
    """
    Basic class for FEM Calculation
    Provides basic functions of doing a
    """

    def __init__(self, mesh):
        """
        Initializer for EFEM class

        Args:
                mesh: dict with "node", "element" "perm"
                        "node": N_node * 2 NDArray
                            contains coordinate information of each node
                        "element": N_elem * 3 NDArray
                            contains the node number contained in each element
                        "perm": N_elem NDArray
                            contains permittivity on each element Default is 1
        """
        # check data structure
        self.config = get_config()
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
        self.perm = 1 / self.config["resistance"]
        self.elem_perm = mesh['perm'] * self.perm
        self.elem_variable = np.zeros(np.shape(self.elem_perm))
        self.elem_param = np.zeros((np.shape(self.elem)[0], 9))  # area, b1, b2, b3, c1, c2, c3, x_average, y_average
        self.electrode_center_list = self.config["electrode_centers"]
        self.electrode_num = len(self.electrode_center_list)
        self.electrode_radius = self.config["electrode_radius"]
        self.electrode_mesh = dict()
        for i in range(self.electrode_num):
            self.electrode_mesh[i] = list()
        self.freq = self.config["signal_frequency"] * 2 * np.pi
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
            electrode_input: Signal input position
        Returns:
            node_potential, element_potential, electrode_potential
        """
        self.calc_init()  # 0.053 s


        # Your Functions Starts Here
        self.my_solver(electrode_input)
        # Your Functions Ends Here


        self.calculate_element_potential()  # 0.004s
        self.calc_electrode_potential()  # 0.001s
        return self.node_potential, self.element_potential, self.electrode_potential

    @abstractmethod
    def my_solver(self, electrode_input):
        """
        REWRITE THIS FOR CUSTOM DIFFERENTIAL FUNCTION

        Do the calculation based on self.elem_param and self.elem_variable and generate the distribution of the
        potential on the surface.
        Edit the self.node_potential vector for final out put.

        Args:
            electrode_input: Signal input position

        Returns:

        """
        pass

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

    def calculate_element_potential(self):
        """
        Get each element's potential,

        Average of each nodes on element
        """
        for i, _ in enumerate(self.elem_param):
            k1, k2, k3 = self.elem[i]
            self.element_potential[i] = (self.node_potential[k1] + self.node_potential[k2] + self.node_potential[
                k3]) / 3

    def calc_electrode_potential(self):
        """
        Get the mean value of potential on every electrode,
        """
        for i, elements in enumerate(self.electrode_mesh.values()):
            potential = []
            for element in elements:
                potential.append(self.element_potential[element])
            self.electrode_potential[i] = np.mean(np.array(potential))

    def plot_variable_map(self, ax):
        """
        Plot the current variable map,

        Args:
            ax: matplotlib.pyplot axis class
        Returns:
            Image
        """
        im = self.plot_map(ax, self.elem_variable)
        return im

    def plot_potential_map(self, ax):
        """
            Plot the current variable map,

            Args:
                ax: matplotlib.pyplot axis class
            Returns:
                Image
        """
        im = self.plot_map(ax, self.element_potential)
        return im

    def change_variable_elementwise(self, element_list, variable_list):
        """Change variable in certain area according to ELEMENT NUMBER,

        Args:
            element_list: INT LIST element numbers to be changed
            variable_list: FLOAT LIST same dimension list for variable on each element included
        Returns:
            NULL
        """
        if len(element_list) == len(variable_list):
            for i, ele_num in enumerate(element_list):
                if ele_num > self.elem_num:
                    raise Exception("Element number exceeds limit")
                self.elem_variable[ele_num] = variable_list[i]
        else:
            raise Exception('The length of element doesn\'t match the length of variable')

    def change_variable_geometry(self, center, radius, value, shape):
        """Change variable in certain area according to GEOMETRY,

        Args:
            center: [FLOAT , FLOAT] center of the shape
            radius: FLOAT radius (half side length) of the shape
            value: FLOAT area variable value
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
                    self.elem_variable[i] = value
                    count += 1
            if count == 0:
                raise Exception("No element is selected, please check the input")
        elif shape == "circle":
            center_x, center_y = center
            count = 0
            for i, x in enumerate(self.elem_param[:, 7]):
                if np.sqrt((center_x - x) ** 2 + (center_y - self.elem_param[i][8]) ** 2) <= radius:
                    self.elem_variable[i] = value
                    count += 1
        else:
            raise Exception("No such shape, please check the input")

    def change_add_variable_geometry(self, center, radius, value, shape):
        """Add variable in certain area according to GEOMETRY

        Args:
            center: [FLOAT , FLOAT] center of the shape
            radius: FLOAT radius (half side length) of the shape
            value: FLOAT area variable value
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
                    self.elem_variable[i] += value
                    count += 1
            if count == 0:
                raise Exception("No element is selected, please check the input")
        elif shape == "circle":
            center_x, center_y = center
            count = 0
            for i, x in enumerate(self.elem_param[:, 7]):
                if np.sqrt((center_x - x) ** 2 + (center_y - self.elem_param[i][8]) ** 2) <= radius:
                    self.elem_variable[i] += value
                    count += 1
        else:
            raise Exception("No such shape, please check the input")

    def reset_variable(self, overall_variable=0):
        """
        Set variable on every value to overall_variable,

        Args:
            overall_variable: FLOAT target variable value for every element
        """
        self.elem_variable = np.zeros(np.shape(self.elem_perm)) + overall_variable

    def reset_variable_to_initial(self, variable_value):
        """
        DEPRECATED
        Set initial distribution of variable density value
        """
        self.elem_variable = np.zeros(np.shape(self.elem_perm))
        self.change_variable_geometry([0, 0], 15, variable_value, shape="square")

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
            raise Exception('The length of element doesn\'t match the length of variable')

    def plot_map(self, ax, param):
        """
        Plot the current variable map,

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

        return im
