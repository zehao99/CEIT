import numpy as np;

node_list = np.linspace(0,99,100)
record = np.array([node_list,node_list])

def swap(node_list, a, b):
    temp = a
    temp = b