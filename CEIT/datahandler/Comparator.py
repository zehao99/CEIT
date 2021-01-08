import matplotlib.pyplot as plt
import numpy as np

from .Data import Data, CalibData, rev_translate_corr
from .DataHandler import DataHandler
from ..Solver import Solver


class Comparator:
    """
    Compare different data points.
    """

    def __init__(self, heights):
        """
        Initializer for comparator object

        Args:
            heights: height list for comparator
        """
        self.heights = heights
        self.solver = Solver()
        self.handler = DataHandler("", solver=self.solver)
        self.h_to_idx = {}
        for i, h in enumerate(heights):
            self.h_to_idx[h] = i

    def compare_and_visualize_two_set_data(self, data_1: Data, calibdata_1: CalibData, data_2: Data,
                                           calibdata_2: CalibData):
        """
        Generate a graph comparing two different conditions.

        Args:
            data_1: data file for condition 1
            calibdata_1: calibration file for condition 1
            data_2: data file for condition 2
            calibdata_2: calibration file for condition 2
        """
        data_1.calc_delta_V(calibdata_1)
        data_2.calc_delta_V(calibdata_2)
        capa_1 = data_1.get_capacitance(self.solver)
        capa_2 = data_2.get_capacitance(self.solver)
        minVal = min(np.min(capa_1), np.min(capa_2))
        maxVal = max(np.max(capa_1), np.max(capa_2))
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
        im1 = self.handler.draw_one_situation(data_1, calibdata_1, ax1, vmax=maxVal, vmin=minVal,
                                              title_prefix="40*40mm at ")
        _ = self.handler.draw_one_situation(data_2, calibdata_2, ax2, vmax=maxVal, vmin=minVal,
                                            title_prefix="80*80mm at ")
        fig.subplots_adjust(right=0.8)
        colorbar_ax = fig.add_axes([0.9, 0.10, 0.03, 0.8])
        fig.colorbar(im1, cax=colorbar_ax)
        plt.show()

    def pick_two_data_and_compare(self, folder_path_1: str, file_name_1: str, calib_file_name_1: str,
                                  folder_path_2: str,
                                  file_name_2: str,
                                  calib_file_name_2: str):
        """
        Pick two data files and generate a comparison graph

        Args:
            folder_path_1: folder path for data1
            file_name_1: filename for data1
            calib_file_name_1: calibration file name for data1
            folder_path_2: folder path for data2
            file_name_2: filename for data2
            calib_file_name_2: calibration file name for data2
        """
        self.handler.set_folder_path(folder_path_1)
        data1 = self.handler.read_one_data_file(file_name_1, contain_excitation=True)
        calib_1 = self.handler.read_one_calibration_file(calib_file_name_1, contain_excitation=True)
        self.handler.set_folder_path(folder_path_2)
        data2 = self.handler.read_one_data_file(file_name_2, contain_excitation=True)
        calib_2 = self.handler.read_one_calibration_file(calib_file_name_2, contain_excitation=True)
        self.compare_and_visualize_two_set_data(data1, calib_1, data2, calib_2)

    def assign_folder_and_file_names(self, smp: str, exp: str, loop_num: int, x: int, y: int, h: int) -> tuple:
        """
        Return the formalized folder path and data file names.

        Args:
            smp: sample number
            exp: experiment number
            loop_num: loop number
            x: x coordinate
            y: y coordinate
            h: h coordinate

        Returns:
            (folder path, calibration filename, data filename)
        """
        x, y = rev_translate_corr(x, y)
        h_idx = self.h_to_idx[h]
        folder = "./data/Smp" + str(smp) + "_" + exp + "/" + str(loop_num)
        calib_name = "calibration_sample" + str(smp) + "_ex" + exp + "0" + str(loop_num) + "_" + str(h_idx) + ".csv"
        data_name = "data_sample" + str(smp) + "_ex" + exp + "0" + str(loop_num) + "_" + str(x) + "_" + str(
            y) + "_" + str(
            self.heights[h_idx]) + ".csv"
        return folder, calib_name, data_name

    def compare_given_coordinate_condition(self, smp1: str, exp1: str, loop_num1: int, x1: int, y1: int, h1: int,
                                           smp2: str,
                                           exp2: str, loop_num2: int, x2: int, y2: int, h2: int):
        """
        Given x, y condition and generate the comparison graph

        Args:
            smp1: sample 1 number
            exp1: experiment num 1
            loop_num1: experiment loop num 1
            x1: x1 coordinate in mm
            y1: y1 coordinate in mm
            h1: height 1 in mm
            smp2: sample 2 number
            exp2: experiment num 2
            loop_num2: experiment loop num 2
            x2: x2 coordinate in mm
            y2: y2 coordinate in mm
            h2: height 2 in mm
        """
        folder_1, calib_name_1, data_name_1 = self.assign_folder_and_file_names(smp1, exp1, loop_num1, x1, y1, h1)
        folder_2, calib_name_2, data_name_2 = self.assign_folder_and_file_names(smp2, exp2, loop_num2, x2, y2, h2)
        self.pick_two_data_and_compare(folder_1, data_name_1, calib_name_1, folder_2, data_name_2, calib_name_2)

    def compare_multiple_set_of_data(self, data: list) -> None:
        if len(data) > 25:
            raise Exception("Too many sets of data (keep it smaller than 25)... Aborting...")
