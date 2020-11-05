# coding=utf-8
import numpy as np
import pickle
from json import loads
import csv
import os

def get_config():
    """
    Get info in the config.json file
    and turn all data to SI units
    """
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.dirname(os.path.dirname(path))
    with open(path+'\\config.json', 'r', encoding='utf-8') as f:
        config = loads(f.read())
    assert config["unit"] == "mm" or config["unit"] == "SI", "Please enter the accurate unit."
    config["rootdir"] = path
    if config["unit"] == "mm":
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
        path_name: path name of the destination example: such as path = os.path.dirname(os.path.realpath(__file__))
    """
    with open(path_name + "\\" + 'cache_' + filename + '.pkl', "wb") as file:
        pickle.dump(param, file)


def read_parameter(filename: object, path_name: object = ".") -> object:
    """
    Read from .pkl file Use this only with the save_parameter() utility !IMPORTANT,

    Args:
        filename: filename of the destination with suffix example: "aaa.csv"
        path_name: absolute path name of the destination example: such as path = os.path.dirname(os.path.realpath(__file__))
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
        path_name: path name of the destination example: "./MESH"
    """
    assert filename.endswith(".csv"), "The filename is not ended with csv."
    data = np.array(data)
    if len(data.shape) > 2:
        raise ValueError("This function can only store two dimension data.")
    with open(path_name + "/" + filename, mode='w', newline='') as file:
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
        path_name: path name of the destination example: "./MESH"
    Returns:
        data: data in the file
    """
    data = []
    assert filename.endswith(".csv"), "The filename is not ended with csv."
    with open(path_name + "/" + filename, newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for line in csv_reader:
            if line:
                line = [float(x) for x in line]
                data.append(line)
    return data
