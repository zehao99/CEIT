from abc import ABCMeta, abstractmethod

import numpy as np

from ..Solver import Solver


class BasicData(metaclass=ABCMeta):
    def __init__(self, data: np.ndarray, electrode_num: int = 16, contain_excitation: bool = False):
        """
        Data Initializer
        Args:
            data: data array
            electrode_num: total electrode number
            contain_excitation: if it contains excitation data
        """
        self.data = data
        self.data_mean = np.mean(self.data, axis=0)
        # self.data_mean = self.data[0, :]
        self.contain_excitation = contain_excitation
        if contain_excitation:
            self.data_mean = self.exclude_excitation(electrode_num)
        self.delta_V = None

    def exclude_excitation(self, electrode_num):
        new_val = []
        for i, val in enumerate(self.data_mean):
            if i % electrode_num != 0:
                new_val.append(val)
        return np.array(new_val)

    @abstractmethod
    def return_copy(self):
        """
        Returns copy of the current data
        """
        pass


class CalibData(BasicData):
    def __init__(self, data, i: int, contain_excitation=False):
        """
        Calibration data initializer
        Args:
            data: data array
            i: height index
            contain_excitation: if it contains excitation data
        """
        super().__init__(data, contain_excitation=contain_excitation)
        self.height_idx = i

    def return_copy(self):
        return CalibData(np.copy(self.data), self.height_idx, contain_excitation=self.contain_excitation)


class Data(BasicData):
    def __init__(self, data: np.ndarray, x: int, y: int, z: int, transformed: bool = False,
                 contain_excitation: bool = False):
        """
        Experiment data initializer

        Args:
            data: Data array
            x: x coordinate
            y: y coordinate
            z: height
            transformed: if the x and y coordinate is transformed to sensor coordinate
            contain_excitation: if it contains excitation data
        """
        if transformed:
            self.x = x
            self.y = y
            self.z = z
        else:
            self.x, self.y = translate_corr(x, y)
            self.z = z
        # self.data_mean = np.mean(self.data, axis = 0)
        self.capacitance_predict = None
        super().__init__(data, contain_excitation=contain_excitation)

    def calc_delta_V(self, calibration: CalibData):
        """
        Calculate delta_V from calibration data

        Args:
            calibration: CalibData object for calculating delta_V
        """
        if len(self.data_mean) != len(calibration.data_mean):
            raise Exception("Shape does not match with calibration.")
        self.delta_V = self.data_mean - np.copy(calibration.data_mean)

    def get_capacitance(self, solver: Solver):
        """
        Get the capacitance value using the solver given.

        Args:
            solver: Solver object for generating capacitance

        Returns:
            NDarray containing capacitance values inside detection area.
        """
        if self.capacitance_predict is not None:
            return np.copy(self.capacitance_predict)
        if self.delta_V is None:
            print(
                "Be aware that you haven't generated delta_V yet, the method would return an empty NDarray")
            return np.array([])
        else:
            self.capacitance_predict = solver.solve(self.delta_V)
            return np.copy(self.capacitance_predict)

    def return_copy(self):
        return Data(np.copy(self.data), self.x, self.y, self.z, transformed=True,
                    contain_excitation=self.contain_excitation)


def translate_corr(x, y):
    return x - 100, -y + 100


def rev_translate_corr(x, y):
    return x + 100, - y + 100
