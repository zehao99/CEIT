from ..readmesh import read_mesh_from_csv
from ..util.utilities import get_config, PointStack, Comp, quicksort
import numpy as np


class MeshObj(object):
    """
    Mesh object for Calculation
    """

    def __init__(self, mesh_obj=None, electrode_num=None, electrode_center_list=None, electrode_radius=None):
        """
        Initialize the mesh object, it will generat the parameter if it's not given
        Args:
            mesh_obj: see readmesh.py
            electrode_num: electrode number
            electrode_center_list: electrode center point list
            electrode_radius: electrode radius(half side length)
        """
        self.electrode_mesh = dict()
        if mesh_obj is None or electrode_num is None or electrode_center_list is None or electrode_radius is None:
            mesh_obj, electrode_num, electrode_center_list, electrode_radius = read_mesh_from_csv().return_mesh()
        self.config = get_config()
        self.mesh_obj = mesh_obj
        self.electrode_num = electrode_num
        self.electrode_center_list = electrode_center_list
        self.electrode_radius = electrode_radius
        self.nodes = self.mesh_obj["node"]
        self.point_x = self.nodes[:, 0]
        self.point_y = self.nodes[:, 1]
        self.elem = self.mesh_obj["element"]
        self.elem_perm = self.mesh_obj["perm"]
        self.detection_index = np.zeros((len(self.elem)))
        self.detection_elem = np.copy(self.elem)
        self.elem_param = np.zeros((np.shape(self.elem)[0], 9))
        for i in range(self.electrode_num):
            self.electrode_mesh[i] = list()
        self.initialize_parameters()
        self.calc_detection_elements()

    def initialize_parameters(self):
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
            # get interpolation parameters
            parameter_mat = np.linalg.inv(parameter_mat)
            parameter_mat = parameter_mat.T
            b = list(parameter_mat[:, 0])
            c = list(parameter_mat[:, 1])
            x_average = np.mean(x)  # get center point coordinate
            y_average = np.mean(y)
            self.elem_param[count] = [area, b[0], b[1],
                                      b[2], c[0], c[1], c[2], x_average, y_average]
            count += 1
        for i in range(self.electrode_num):
            center = self.electrode_center_list[i]
            self.calc_electrode_elements(i, center, self.electrode_radius)

    def calc_electrode_elements(self, electrode_number, center, radius):
        """
        Get the electrode element sets for every electrode,

        According to the SQUARE area given and put values into electrode_mesh dict

        Args:
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
                raise Exception(
                    "No element is selected, please check the input")

    def return_mesh(self):
        """
        Return the mesh dict

        Returns:
            dict: mesh info.
        """
        return self.mesh_obj

    def return_electrode_info(self):
        return self.electrode_center_list, self.electrode_radius

    def calc_detection_elements(self):
        """
        Get elements whose center is inside detection range and not inside electrode,
        """
        original_element = self.elem
        corres_index = []
        new_elem = []
        # Flatten out the mesh inside electrode
        flattened_electrode_elem = set()
        for elems in self.electrode_mesh.values():
            for elem in elems:
                flattened_electrode_elem.add(elem)
        for i, element in enumerate(original_element):
            x_val = 0
            y_val = 0
            for idx in element:
                x_val += self.nodes[idx][0]
                y_val += self.nodes[idx][1]
            x_val /= 3
            y_val /= 3
            # filter out mesh outside detection range and on the electrodes.
            if i not in flattened_electrode_elem and np.abs(x_val) < self.config["detection_bound"] and np.abs(y_val) < \
                    self.config["detection_bound"]:
                corres_index.append(i)
                new_elem.append(element)
        self.detection_index = np.array(corres_index)
        self.detection_elem = np.array(new_elem)

    def delete_outside_detect(self, list_c):
        """
        Delete the elements inside a list corresponding to the elements inside detection

        Args:
            list_c : an all-element-wise list

        Returns:
             elements remained in detection domain
        """
        list_c = np.array(list_c)
        if list_c.ndim > 1:
            new_list_c = np.zeros(
                (self.detection_index.shape[0], list_c.shape[1]))
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

    def get_perimeter(self):
        """
        Get the perimiter of the mesh, return in idx sequence.

        This function get the Convex Hull of the mesh and return the chain of node index list of the hull.

        Returns:
            node index list on perimeter in CCW.
        """
        # get start point
        start_point = 0
        min_y = 100000000
        x = 100000000
        for i, node in enumerate(self.nodes):
            if node[1] < min_y:
                start_point = i
                min_y = node[1]
                x = node[0]
            elif node[1] == min_y:
                if node[0] < x:
                    start_point = i
                    x = node[0]
        # sort the rest points
        nodes = np.zeros((self.nodes.shape[0] - 1, self.nodes.shape[1] + 1))
        pointer = 0
        for i, node in enumerate(self.nodes):
            if i != start_point:
                nodes[pointer] = np.array([node[0], node[1], i])
                pointer += 1
        comp = Comp(self.nodes[start_point])
        quicksort(nodes, comp)
        # construct perimeter
        stack = PointStack()
        orig = np.array([self.nodes[start_point][0], self.nodes[start_point][1], start_point])
        stack.push(orig)
        stack.push(nodes[0])
        stack.push(nodes[1])
        for i, node in enumerate(nodes):
            if i < 2:
                continue
            while len(stack) > 1 and ccw(stack.next_to_top(), stack.peek(), node) in [-1, 0]:
                a = stack.next_to_top()
                b = stack.peek()
                c = node
                ans = ccw(a, b, c)
                stack.pop()
            stack.push(node)
        # prepare return array
        ans = np.zeros((len(stack)), dtype=np.int)
        pointer = len(stack) - 1
        while not stack.isEmpty():
            idx = int(stack.pop()[2])
            ans[pointer] = idx
            pointer -= 1
        return ans


def ccw(p0, p1, p2):
    """
    Judge whether p0p2 vector is ccw to p0p1 vector.

    Return value map: \n
        1: p0p2 is ccw to p0p1 (angle to x axis bigger) \n
        0: p0p2 and p0p1 on a same line \n
        -1: p0p2 is cw to p0p1 (angle to x axis smaller) \n

    Args:
        p0: base point index 0 and 1 is x and y value. [x, y, ...]
        p1: first point index 0 and 1 is x and y value. [x, y, ...]
        p2: second point index 0 and 1 is x and y value. [x, y, ...]

    Returns:
        int: judgement value -1 or 0 or 1
    """
    comp = Comp(p0)
    return comp.compare_angle(p2, p1)
