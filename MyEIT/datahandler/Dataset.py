import os

import matplotlib.pyplot as plt
import numpy as np

from .Data import rev_translate_corr, CalibData, Data
from .DataHandler import DataHandler


class DataSet:
    """
    Dataset which holds data for one set of experiment data

    Attributes:
        handler: data handler for single file
        calibration_data: calibration data dict key is height and value is CalibData object
        dataset: dataset of
    """

    def __init__(self, smp: str, exp: str, loop_num: int, height_list: list, initialize=True):
        """
        Initializer for dataset

        Your data folder should structure like following:
        |-./                      # root folder
          |-/data
            |-/Smp01_001         # experiment 001 for sample 01
              |-/0               # loop number of your experiment

        Every set of data contains of two parts, one calibration data and other data

        for calibration data, the filename should be like:
            Example: calibration_sample01_ex00100_0.csv

            which means calibration data for height index 0 inside self.height_list inside experiment 001 loop 0
            and sample number is 01

        for record data, the file name should be like:
            Example: data_sample08_ex00100_20_20_50.csv

            which means data at point (20, 20) mm height 50mm in experiment 001 loop 0 and sample number is 01

        Args:
            smp: Sample number in string
            exp: experiment number in string
            loop_num: int loop number
            height_list: list of heights
            initialize: whether read all the files or not
        """
        self.sample_num, self.exp, self.loop_num = smp, exp, loop_num
        self.height_list = height_list
        self.height_to_idx = {}
        for i, h in enumerate(height_list):
            self.height_to_idx[h] = i
        self.folder_path = "./data/Smp" + self.sample_num + "_" + self.exp + "/" + str(self.loop_num)
        self.dataset = {}
        self.calibration_data = {}
        self.handler = DataHandler(self.folder_path)
        for h in self.height_list:
            self.dataset[h] = []
        if initialize:
            self.read_all_files_in_folder()

    def read_all_files_in_folder(self) -> None:
        """
        Read all files inside the folder
        """
        data_file_names = []
        for i, j, k in os.walk(self.folder_path):
            data_file_names = k
        for name in data_file_names:
            if name.split('_')[0] == "calibration":
                onedata = self.handler.read_one_calibration_file(name, contain_excitation=True)
                if onedata.height_idx > len(self.height_list) - 1:
                    raise Exception("The height list provided doesn't fit the data. Aborting...")
                self.calibration_data[self.height_list[onedata.height_idx]] = onedata.return_copy()
            else:
                onedata = self.handler.read_one_data_file(name, contain_excitation=True)
                if onedata.z not in self.height_to_idx.keys():
                    raise Exception("The height list provided doesn't fit the data. Aborting...")
                self.dataset[onedata.z].append(onedata.return_copy())

    def __getitem__(self, height: int) -> dict:
        """
        Return a dict of data at certain height

        Args:
            height: height value

        Returns:
             {"calibration_data": calibration_data, "dataset": dataset at height}
        """
        if len(self.dataset.keys()) == 0:
            raise Exception(
                "Please call read_all_files_in_folder() function on the object to get data or you have an empty folder")
        if height not in self.height_list:
            raise IndexError("Height not inside height list")
        return {"calibration_data": self.calibration_data[height], "dataset": self.dataset[height]}

    def get_data_at(self, x: int, y: int, height: int) -> Data:
        """
        Return data at certain point

        Args:
            x: x coordinate
            y: y coordinate
            height: height value

        Returns:
            Data: data object
        """
        data_dict = self.__getitem__(height)
        for data in data_dict["dataset"]:
            if data.x == x and data.y == y:
                return data.return_copy()
        raise IndexError("x and y not inside dataset")

    def get_calibration_data_at(self, height: int) -> CalibData:
        """
        Get calibration data at height from the dataset

        Args:
            height: int

        Returns:
            Calibration data at height
        """
        data_dict = self.__getitem__(height)
        return data_dict["calibration_data"].return_copy()

    def return_path_for_specific_point(self, x: int, y: int, h) -> tuple:
        """
        return file names for a single position.
        Args:
            x: x coordinate
            y: y coordinate
            h: z coordinate

        Returns:
            tuple: (folder path, calib file name, data file name)

        """
        h_idx = self.height_to_idx[h]
        x, y = rev_translate_corr(x, y)
        calib_name = "calibration_sample" + self.sample_num + "_ex" + self.exp + "0" + str(self.loop_num) + "_" + str(
            h_idx) + ".csv"
        data_name = "data_sample" + self.sample_num + "_ex" + self.exp + "0" + str(self.loop_num) + "_" + str(
            x) + "_" + str(y) + "_" + str(
            self.height_list[h_idx]) + ".csv"
        return self.folder_path, calib_name, data_name

    def draw_full_COP_map_for_every_height(self) -> None:
        """
        Draw the full COP Mappings

        Returns:
            None
        """
        fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), ax_long
              ) = plt.subplots(ncols=5, nrows=3)
        row = ax_long[0].get_gridspec()
        for ax in ax_long:
            ax.remove()
        amp_ax = fig.add_subplot(row[-1, 0:])

        axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
        vals = []
        for i, h in enumerate(self.height_list):
            vals.append(self.handler.draw_COP(self.dataset[h], self.calibration_data[h], axs[i], "H " + str(h) + "mm"))

        labels = []
        for data in self.dataset[50]:
            labels.append('(' + str(data.x) + ',' + str(data.y) + ')')
        x = np.arange(len(vals[0]))

        labels.append('')
        labels.append('')
        labels.append('')
        width = 0.05
        for i, val in enumerate(vals):
            amp_ax.bar(x + width * (i - (len(vals) - 1) / 2), val, width, label=str(self.height_list[i]) + 'mm')
        x = np.arange(len(vals[0]) + 3)
        amp_ax.set_ylabel('Amplitude(F/m^2)')
        amp_ax.set_xticks(x)
        amp_ax.set_xticklabels(labels)
        amp_ax.set_title('Capacitance Max Value on every position')
        amp_ax.legend()

        plt.show()
