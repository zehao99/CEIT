import csv

import matplotlib.pyplot as plt
import numpy as np

from .Data import Data, CalibData
from ..Solver import reinitialize_solver, Solver
from ..util.utilities import get_config

REGULARIZATION_PARAM = get_config()["regularization_coeff"]
THRESHOLD_PERCENTAGE = 0.9


class DataPlotter:
    """
    Plotter for data.

    Attributes:
        solver: Solver object for inverse problems.
    """

    def __init__(self, solver: Solver = None):
        if solver is None:
            self.solver = reinitialize_solver(REGULARIZATION_PARAM)
        else:
            self.solver = solver

    def get_COP(self, data: Data):
        """
        Gets the center of position of the given data

        Args:
            data: Data class defined in {@class Data.py}

        Returns:
            x and y coordinates of the center and mean value bigger than Percentage threshold of the whole mesh
        """
        if data.delta_V is None:
            raise Exception("WRONG!")
        capacitance = self.solver.solve(data.delta_V)
        x_sum = 0.0
        y_sum = 0.0
        v_sum = 0.0
        count = 0.0
        threshold = np.max(capacitance) * THRESHOLD_PERCENTAGE + np.min(capacitance) * (1 - THRESHOLD_PERCENTAGE)
        for i, c in enumerate(capacitance):
            if c > threshold:
                idx = self.solver.mesh.detection_index[i]
                x_sum += (self.solver.mesh.elem_param[idx][7])
                y_sum += (self.solver.mesh.elem_param[idx][8])
                v_sum += c
                count += 1
        x_mean = x_sum / count
        y_mean = y_sum / count
        v_mean = v_sum / count
        return x_mean, y_mean, v_mean

    def draw_COP(self, dataset, calibration, ax, title="COP"):
        """
        Draw a COP map given the dataset.

        Args:
            dataset: list of Data object
            calibration: CalibData Object for calibration
            ax: matplotlib.pyplot.axes object
            title: title of the graph

        Returns:
            values: mean value after threshold for every situation inside dataset list.
        """
        x_re = []
        y_re = []
        x_gt = []
        y_gt = []
        values = []
        for data in dataset:
            data.calc_delta_V(calibration)
            x, y, maxVal = self.get_COP(data)
            x_re.append(x * 2000)
            y_re.append(y * 2000)
            x_gt.append(data.x)
            y_gt.append(data.y)
            values.append(maxVal)
            ax.plot([x * 2000, data.x], [y * 2000, data.y],
                    color=(0.1, 0.2, 0.5, 0.2))
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.scatter(x_re, y_re, label="reconstruction")
        ax.scatter(x_gt, y_gt, label="Ground Truth")
        ax.set_title(title)
        # ax.legend()
        ax.set_aspect('equal')
        ax.grid(True)
        return np.array(values)

    def draw_one_situation(self, data: Data, calibration: CalibData, ax: plt.axes, vmax=None, vmin=None,
                           title_prefix=""):
        """
        Draw one situation with object on the graph.

        Args:
            data: Data object to be visualized
            calibration: CalibData object used for calibration
            ax: matplotlib.pyplot.axes object
            vmax: max value limit of the graph
            vmin: min value limit of the graph
            title_prefix: prefix for graph title

        Returns:
            im: matplotlib.image object
        """
        obj = data
        obj.calc_delta_V(calibration)
        print(obj.x, obj.y, obj.z)
        capacitance = self.solver.solve(obj.delta_V)
        e_x, e_y, _ = self.get_COP(obj)
        im = self.solver.plot_map_in_detection_range(ax, capacitance, vmax=vmax, vmin=vmin)
        circle = plt.Circle((obj.x, obj.y),
                            13.5, color='#F55400', fill=False)
        circle_e = plt.Circle((e_x * 2000, e_y * 2000),
                              13.5, color='k', fill=False)
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.add_artist(circle)
        ax.add_artist(circle_e)
        ax.set_title(title_prefix + "(" + ",".join((str(obj.x), str(obj.y), str(obj.z))) + ")")
        ax.set_aspect('equal')
        return im


class DataHandler(DataPlotter):
    """
    Data handler for handling recorded data on different positions.

    For the accurate folder structure, please check README.md or Dataset initializer.

    Attributes:
        folder_path: path to the data folder
    """

    def __init__(self, folder_path: str, solver: Solver = None):
        """
        Initialize the data handler

        Args:
            folder_path: folder path to the data files
        """
        super().__init__(solver)
        self.folder_path = folder_path

    def set_folder_path(self, folder_path: str):
        self.folder_path = folder_path

    def read_from_csv(self, filename):
        data = []
        with open(self.folder_path + "/" + filename) as f:
            csvFile = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            for line in csvFile:
                data.append(line)
        return np.array(data)

    def read_one_calibration_file(self, filename: str, contain_excitation=False) -> CalibData:
        """
        Get one Calibration data object from file

        Args:
            filename: filename of the data
            contain_excitation: if the calibration contains excitation electrode data

        Returns:
            CalibData: calibration data
        """
        data = self.read_from_csv(filename)
        height_idx = get_height_idx_from_filename(filename)
        return CalibData(data, height_idx, contain_excitation=contain_excitation)

    def read_one_data_file(self, filename: str, contain_excitation=False) -> Data:
        """
        Get one reading data object from file

        Args:
            filename: filename of the data
            contain_excitation: if the data contains excitation electrode data

        Returns:
            Data: data object read
        """
        x, y, z = get_corr_from_filename(filename)
        data = self.read_from_csv(filename)
        return Data(data, x, y, z, contain_excitation=contain_excitation)


def get_corr_from_filename(filename: str):
    filename_raw = filename.split('.')[0]
    filename_arr = filename_raw.split('_')
    return int(filename_arr[3]), int(filename_arr[4]), int(filename_arr[5])


def get_height_idx_from_filename(filename: str):
    filename_raw = filename.split('.')[0]
    return int(filename_raw.split('_')[-1])
