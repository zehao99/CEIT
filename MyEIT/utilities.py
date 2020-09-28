# coding=utf-8
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pickle
from json import loads


def get_config():
    with open('./config.json', 'r', encoding='utf-8') as f: 
        config = loads(f.read())
    return config


def save_parameter(param, filename, folder_name = "."):
    with open(folder_name +"/" + 'cache_'+ filename +'.pkl', "wb") as file:
        pickle.dump(param, file)


def read_parameter(filename, folder_name = "."):
    with open(folder_name +"/" + 'cache_'+ filename +'.pkl', 'rb') as file:
        param = pickle.load(file)
    return param
