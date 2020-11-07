# coding=utf-8
import numpy as np
import pickle
from json import loads
import csv
import os
import math


def get_config():
    """
    Get info in the config.json file
    and turn all data to SI units

    Returns:
        dict: config info
    """
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.dirname(os.path.dirname(path))
    with open(path + '\\config.json', 'r', encoding='utf-8') as f:
        config = loads(f.read())
    assert config["sensor_param_unit"] == "mm" or config["sensor_param_unit"] == "SI", "Please enter the accurate unit."
    config["rootdir"] = path
    if config["sensor_param_unit"] == "mm":
        config["electrode_centers"] = list(np.array(config["electrode_centers"]) / 1000)
        config["electrode_radius"] = config["electrode_radius"] / 1000
        config["detection_bound"] = config["detection_bound"] / 1000
    return config


def save_parameter(param, filename, path_name="."):
    """
    Save parameter to .pkl file,

    Args:
        param: parameter to save
        filename: filename of the destination without suffix ".pkl" example: "mesh_cache"
        path_name: path name of the destination example:
                    such as path = os.path.dirname(os.path.realpath(__file__))
    """
    with open(path_name + "\\" + 'cache_' + filename + '.pkl', "wb") as file:
        pickle.dump(param, file)


def read_parameter(filename, path_name):
    """
    Read from .pkl file Use this only with the save_parameter() utility !IMPORTANT,

    Args:
        filename: filename of the destination with suffix example: "aaa.csv"
        path_name: absolute path name of the destination example:
                    such as path = os.path.dirname(os.path.realpath(__file__))
    Returns:
        data: data in the file
    """
    with open(path_name + "\\" + 'cache_' + filename + '.pkl', 'rb') as file:
        param = pickle.load(file)
    return param


def save_to_csv_file(data, filename, path_name="."):
    """
    Save parameter to .csv file,

    Args:
        data: parameter to save must be 1D or 2D data
        filename: filename of the destination with suffix example: "aaa.csv"
        path_name: absolute path name of the destination example:
                    such as path = os.path.dirname(os.path.realpath(__file__))
    """
    assert filename.endswith(".csv"), "The filename is not ended with csv."
    data = np.array(data)
    if len(data.shape) > 2:
        raise ValueError("This function can only store two dimension data.")
    with open(path_name + "\\" + filename, mode='w', newline='') as file:
        data_writer = csv.writer(file, delimiter=',')
        if len(data.shape) == 1:
            data_writer.writerow(data)
        else:
            for line in data:
                data_writer.writerow(line)


def read_csv_from_file(filename, path_name="./"):
    """
    Read from .csv file

    Args:
        filename: filename of the destination with suffix example: "aaa.csv"
        path_name: absolute path name of the destination example:
                    such as path = os.path.dirname(os.path.realpath(__file__))
    Returns:
        data: data in the file
    """
    data = []
    assert filename.endswith(".csv"), "The filename is not ended with csv."
    with open(path_name + "\\" + filename, newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for line in csv_reader:
            if line:
                line = [float(x) for x in line]
                data.append(line)
    return data


def read_csv_one_line_from_file(filename, path_name=".", idx=0):
    """
    Read from .csv file one line

    Args:
        filename: filename of the destination with suffix example: "aaa.csv"
        path_name: absolute path name of the destination example:
                    such as path = os.path.dirname(os.path.realpath(__file__))
        idx: line of the data, default to 0
    Returns:
        data: data in the file
    """
    data = []
    assert filename.endswith(".csv"), "The filename is not ended with csv."
    with open(path_name + "\\" + filename, newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')
        count = 0
        for line in csv_reader:
            if line:
                if count == idx:
                    data = [float(x) for x in line]
                    break
                else:
                    count += 1
    return data


def get_angle(p1, p2):
    """
    Get the angle of vector from p1 to p2.
    Args:
        p1: point 1 example : [x1, y1]
        p2: point 2 example : [x2, y2]
    Return:
        angle of the vector from 0 to 2pi.
    """
    if p1[0] == p2[0]:
        if p1[1] <= p2[1]:
            return math.pi / 2
        else:
            return math.pi * 3 / 2
    if p1[1] == p2[1]:
        if p1[0] <= p2[0]:
            return 0
        else:
            return math.pi
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    if delta_x > 0:
        if delta_y > 0:
            return np.arctan(delta_x / delta_y)
        else:
            return np.arctan(delta_x / delta_y) + math.pi * 2
    else:
        if delta_y > 0:
            return np.arctan(delta_x / delta_y) + math.pi
        else:
            return np.arctan(delta_x / delta_y) + math.pi


class Comp:
    """
    Comparator for direction between three points

    Takes in two points p1, p2 and compare the relationship between vector p0p1 and p0p2
    """
    def __init__(self, p0):
        """
        Initialize the comparator

        Args:
            p0: ground point of comparator
        """
        self.p0 = np.copy(p0)

    def compare(self, p1, p2):
        """
        Compare two points with respect to the original point p0

        Return value map: \n
        1: p0p1 is ccw to p0p2 (angle to x axis bigger) or on same line but p1 is further \n
        0: p0p1 and p0p2 on a same line \n
        -1: p0p1 is cw to p0p2 (angle to x axis smaller) or on same line but p1 is closer \n

        Args:
            p1: first point x, y axis value at index 0 and 1
            p2: second point x, y axis value at index 0 and 1

        Returns:
            int: judgement value -1 or 0 or 1
        """
        ans = (p1[1] - self.p0[1]) * (p2[0] - self.p0[0]) - (p1[0] - self.p0[0]) * (p2[1] - self.p0[1])
        if ans < -1e-10:
            return -1
        elif ans > 1e-10:
            return 1
        else:
            return distance_2D(p1, self.p0) - distance_2D(p2, self.p0)

    def compare_angle(self, p1, p2):
        """
        Only compare angle between two vectors.

        Return value map: \n
        1: p0p1 is ccw to p0p2 (angle to x axis bigger) \n
        0: p0p1 and p0p2 on a same line \n
        -1: p0p1 is cw to p0p2 (angle to x axis smaller) \n

        Args:
            p1: first point x, y axis value at index 0 and 1
            p2: second point x, y axis value at index 0 and 1

        Returns:
            int: judgement value -1 or 0 or 1
        """
        ans = (p1[1] - self.p0[1]) * (p2[0] - self.p0[0]) - (p1[0] - self.p0[0]) * (p2[1] - self.p0[1])
        if ans < -1e-10:
            return -1
        elif ans > 1e-10:
            return 1
        else:
            return 0


def distance_2D(p1, p2):
    """
    Get distance between two points

    Args:
        p1: first point index 0 and 1 should be x and y axis value
        p2: second point index 0 and 1 should be x and y axis value

    Returns:
        distance between two points
    """
    return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])


class PointStack:
    """
    Stack for points to find Convex hull
    """

    def __init__(self, array=None):
        """
        Args:
            array: initialize array, 2D array of points example: [[0, 0, 1]]
        """
        self.stack = []
        if array is not None:
            for item in array:
                self.stack.append(item)

    def __len__(self):
        return len(self.stack)

    def isEmpty(self):
        """

        Returns:
            bool wheter the stack is empty or not.
        """
        return len(self.stack) <= 0

    def push(self, item):
        """
        Add item to the Stack

        Args:
             item: item to be added
        """
        self.stack.append(item)

    def pop(self):
        """
        Pop the top element from the stack

        Returns:
            NdArray copy of poped item
        """
        temp = np.copy(self.stack[-1])
        del self.stack[-1]
        return temp

    def next_to_top(self):
        """
        Get the second top element from the stack.

        Returns:
            References of the second top element
        """
        if len(self.stack) < 2:
            raise Exception("No second top element.")
        return self.stack[-2]

    def peek(self):
        """
        Returns:
            reference of top element of the stack
        """
        return self.stack[-1]


def quicksort(array, comp):
    """
    Quick sort the array with comp.compare(a, b) method

    smaller items are at the front \n
    comp.compare method should return the followings: \n
    a > b : return val > 0 \n
    a = b : return val = 0 \n
    a < b : return val < 0 \n

    Args:
        array: List to be sorted
        comp: comparor with comp(a, b) method
    """
    sort_helper(array, comp, 0, len(array) - 1)


def sort_helper(array, comp, start, end):
    """
    Helper function for quicksort()

    Args:
        array: List to be sorted
        comp: comparor with comp(a, b) method
        start: starting index
        end: ending index
    """
    lo = start
    hi = end
    if lo >= hi: return
    p = lo
    lo = lo + 1
    while lo <= hi:
        if comp.compare(array[p], array[hi]) > 0 and comp.compare(array[p], array[lo]) < 0:
            swap(array, lo, hi)
        if comp.compare(array[p], array[hi]) <= 0:
            hi = hi - 1
        if comp.compare(array[p], array[lo]) >= 0:
            lo = lo + 1
    swap(array, p, hi)
    if len(array) - lo > hi - 1:
        sort_helper(array, comp, start, hi - 1)
        sort_helper(array, comp, hi + 1, end)
    else:
        sort_helper(array, comp, hi + 1, end)
        sort_helper(array, comp, start, hi - 1)


def swap(array, idx1, idx2):
    """
    Swap two elements inside a array

    Args:
        array: List to swap
        idx1: first position
        idx2: second position
    """
    temp = np.copy(array[idx1])
    array[idx1] = np.copy(array[idx2])
    array[idx2] = temp
