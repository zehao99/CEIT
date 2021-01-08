from .util.utilities import get_config
import matplotlib.patches as patches
import numpy as np
from abc import ABCMeta, abstractmethod
from .models.mesh import MeshObj
from .EITPlotter import EITPlotter


class FEMBasic(metaclass=ABCMeta):
    """
    Basic class for FEM Calculation

    Provides basic functions of doing a forward calculation.

    Functions:
                calculation(self,electrode_input): Forward Calculation

                plot_potential_map: Plot the forward result

                plot_current_variable: Plot the variable ground truth

                change_variable_elementwise(self, element_list, variable_list): Change variable on a set of elements

                change_variable_geometry(self, center, radius, value, shape): Change variable inside geometric shape

                change_add_variable_geometry(self, center, radius, value, shape): Adding variable to region

                change_conductivity(self, element_list, resistance_list): Change conductivity

                reset_variable(self, overall_variable):Reset all variable to 0

                reset_variable_to_initial(self, variable_value): DEPRECATED Reset with radius 20 square value 10e-8.46
    """

    def __init__(self, mesh=None):
        """
        Initializer for EFEM class

        Args:
                mesh: MeshObj class
        """
        # check data structure
        self.config = get_config()
        if mesh is None:
            self.mesh = MeshObj()
        else:
            self.mesh = mesh
        self.node_num = np.shape(self.mesh.nodes)[0]
        if np.shape(self.mesh.nodes)[1] != 2:
            raise Exception("2D Node coordinates incorrect")
        self.node_num_bound = 0
        self.node_num_f = 0
        self.node_num_ground = 0
        if np.shape(self.mesh.elem)[1] != 3:
            raise Exception("2D Elements dimension incorrect")
        self.elem_num = np.shape(self.mesh.elem)[0]
        self.perm = 1 / self.config["resistance"]
        self.mesh.elem_perm = self.mesh.elem_perm * self.perm
        self.elem_variable = np.zeros(np.shape(self.mesh.elem_perm))
        self.freq = self.config["signal_frequency"] * 2 * np.pi
        self.K_sparse = np.zeros((self.node_num, self.node_num), dtype=np.complex128)
        self.K_node_num_list = [x for x in range(self.node_num)]  # Node number mapping list when calculating
        self.node_potential = np.zeros((self.node_num), dtype=np.complex128)
        self.element_potential = np.zeros((self.elem_num), dtype=np.complex128)
        self.electrode_potential = np.zeros((self.mesh.electrode_num), dtype=np.complex128)

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
        self.electrode_potential = np.zeros((self.mesh.electrode_num), dtype=np.complex128)



    def calculate_element_potential(self):
        """
        Get each element's potential,

        Average of each nodes on element
        """
        for i, _ in enumerate(self.mesh.elem_param):
            k1, k2, k3 = self.mesh.elem[i]
            self.element_potential[i] = (self.node_potential[k1] + self.node_potential[k2] + self.node_potential[
                k3]) / 3

    def calc_electrode_potential(self):
        """
        Get the mean value of potential on every electrode,
        """
        for i, elements in enumerate(self.mesh.electrode_mesh.values()):
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
            for i, x in enumerate(self.mesh.elem_param[:, 7]):
                if (center_x + radius) >= x >= (center_x - radius) and (
                        center_y + radius) >= self.mesh.elem_param[i][8] >= (center_y - radius):
                    self.elem_variable[i] = value
                    count += 1
            if count == 0:
                raise Exception("No element is selected, please check the input")
        elif shape == "circle":
            center_x, center_y = center
            count = 0
            for i, x in enumerate(self.mesh.elem_param[:, 7]):
                if np.sqrt((center_x - x) ** 2 + (center_y - self.elem_param[i][8]) ** 2) <= radius:
                    self.mesh.elem_variable[i] = value
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
            for i, x in enumerate(self.mesh.elem_param[:, 7]):
                if (center_x + radius) >= x >= (center_x - radius) and (
                        center_y + radius) >= self.mesh.elem_param[i][8] >= (center_y - radius):
                    self.elem_variable[i] += value
                    count += 1
            if count == 0:
                raise Exception("No element is selected, please check the input")
        elif shape == "circle":
            center_x, center_y = center
            count = 0
            for i, x in enumerate(self.mesh.elem_param[:, 7]):
                if np.sqrt((center_x - x) ** 2 + (center_y - self.mesh.elem_param[i][8]) ** 2) <= radius:
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
        self.elem_variable = np.zeros(np.shape(self.mesh.elem_perm)) + overall_variable

    def change_conductivity(self, element_list, resistance_list):
        """
        Change conductivity in certain area according to ELEMENT NUMBER

         Args:
            element_list: INT LIST element numbers to be changed
            resistance_list: FLOAT LIST same dimension list for conductivity on each element included
        """
        self.mesh.change_conductivity(element_list, resistance_list)

    def plot_map(self, ax, param):
        """
        Plot the current variable map,

        Args:
            ax: matplotlib.pyplot axis class
            param: parameter to be plotted(must match with the element)
        Returns:
            NULL
        """
        plotter = EITPlotter(self.mesh)
        im = plotter.plot_full_area_map(param, ax)

        return im
