# @author: Li Zehao <https://philipli.art>
# @license: MIT
import csv
import re

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .util.utilities import get_config
from .util.utilities import read_parameter
from .util.utilities import save_parameter

"""
    Call function init_mesh() to initiate mesh after making a new one
"""


def string_to_float(input_num):
    """
    Turn string 0.0000e000 to float,
    """
    if re.match(r'[0-9\.]+\-[0-9]+', input_num):
        num, power = input_num.split('-')
        out = float(num) * 10 ** (-int(power))
    elif re.match(r'\-[0-9\.]+\-[0-9]+', input_num):
        _, num, power = input_num.split('-')
        out = - float(num) * 10 ** (-int(power))
    else:
        out = float(input_num)
    return out


class ReadMesh(object):
    """
    DO NOT DIRECTLY USE THIS CLASS

    Read .fem file
    And assign electrode centers and radius
    Change the electrode shape according to the mesh
    """

    def __init__(self, filename, electrode_centers, electrode_radius, folder_name="", optimize_node_num=False,
                 shuffle_element=False, is_SI=False):
        self.config = get_config()
        self.nodes = []
        self.elements = []
        self.folder_name = folder_name
        with open(self.config["rootdir"] + "\\" + self.config["folder_name"] + "\\" + filename, newline="") as file:
            for line in file:
                if line.find("GRID") != -1 and line.find("$") == -1:
                    x = string_to_float(line[24:32].rstrip())
                    y = string_to_float(line[32:40].rstrip())
                    self.nodes.append([x, y])
                if line.find("CTRIA3") != -1 and line.find("$") == -1:
                    elem_nodes = [0, 0, 0]
                    elem_nodes[0] = int(line[24:32].lstrip()) - 1
                    elem_nodes[1] = int(line[32:40].lstrip()) - 1
                    elem_nodes[2] = int(line[40:48].lstrip()) - 1
                    self.elements.append(elem_nodes)
        # electrode info center, radius
        self.electrode_centers = electrode_centers
        self.electrode_num = len(self.electrode_centers)
        self.electrode_radius = electrode_radius  # end of electrode info
        if not is_SI:
            # turn mm unit to SI unit
            self.nodes = list(np.array(self.nodes) / 1000)
        self.clean_mesh()
        if optimize_node_num:
            self.optimize_node_number()
        if shuffle_element:
            self.shuffle_elements()
        self.output()
        self.out_2pkl()
        print('CSV and PKL file is ready. Mesh parsed.')

    def clean_mesh(self):
        """
        Get rid of repeated nodes
        """
        origin_nodes = self.nodes
        new_nodes = []
        record = []
        for i, node in enumerate(origin_nodes):
            if i in record:
                continue
            current = [i]
            for j, node_2 in enumerate(origin_nodes):
                if node[0] == node_2[0] and node[1] == node_2[1] and i < j:
                    record.append(j)
                    current.append(j)
            new_nodes.append(node)
            index = len(new_nodes) - 1
            for i_1, element in enumerate(self.elements):
                for j_1, node in enumerate(element):
                    if node in current:
                        self.elements[i_1][j_1] = index
        self.nodes = new_nodes

    def return_mesh(self):
        element_num = len(self.elements)
        mesh_obj = {'element': np.array(self.elements), 'node': np.array(self.nodes), 'perm': np.ones(element_num)}
        return mesh_obj, self.electrode_num, self.electrode_centers, self.electrode_radius

    def optimize_node_number(self):
        """
        optimize nodes numbers
        """
        new_node_num = np.zeros((len(self.nodes)), dtype=int)
        polar_list = np.zeros((len(self.nodes), 2))
        for i, node in enumerate(self.nodes):
            polar_list[i][0] = np.sqrt(node[0] ** 2 + node[1] ** 2)
            polar_list[i][1] = self.arc2tan(node[0], node[1], polar_list[i][0])
        p_max = np.max(polar_list[:, 0])
        steps = int(len(self.nodes) / 10)
        p_inc = p_max / steps
        count = 0
        for i in range(steps + 1):
            sort_list = []
            for j, polar in enumerate(polar_list):
                if (p_max - i * p_inc) >= polar[0] > (p_max - (i + 1) * p_inc):
                    sort_list.append([j, polar[1]])
            sort_list = sorted(sort_list, key=(lambda x: x[1]))
            for k, _ in sort_list:
                new_node_num[k] = count
                count += 1
        new_elements = np.zeros((len(self.elements), 3), dtype=int)
        new_nodes = np.zeros((len(self.nodes), 2))
        for i, element in enumerate(self.elements):
            for j, elem_node in enumerate(element):
                new_elements[i][j] = new_node_num[elem_node]
        for i, node in enumerate(self.nodes):
            index = new_node_num[i]
            new_nodes[index][0] = node[0]
            new_nodes[index][1] = node[1]
        self.nodes = new_nodes
        self.elements = new_elements

    def shuffle_elements(self):
        """
        shuffle element numbers
        """
        N = len(self.elements)
        for i in range(N):
            r = np.random.random_integers(0, i)
            swap = np.copy(self.elements[r])
            self.elements[r] = np.copy(self.elements[i])
            self.elements[i] = np.copy(swap)

    def arc2tan(self, x, y, r):
        if y >= 0:
            return np.arccos(x / r)
        if y < 0:
            return - np.arccos(x / r) + 2 * np.pi

    def output(self, file_name='Mesh_Cache'):
        with open(self.config["rootdir"] + "\\" + self.config["folder_name"] + "\\" + file_name + '_Node.csv', mode='w', newline='') as outfile:
            data_writer = csv.writer(outfile, delimiter=',')
            for node in self.nodes:
                data_writer.writerow(node)
        with open(self.config["rootdir"] + "\\" + self.config["folder_name"] + "\\" + file_name + '_Element.csv', mode='w', newline='') as outfile_2:
            data_writer_2 = csv.writer(outfile_2, delimiter=',')
            for element in self.elements:
                data_writer_2.writerow(element)

    def out_2pkl(self, filename='Mesh_'):
        save_parameter(self.nodes, filename + 'nodes', self.config["rootdir"] + "\\" + self.config["folder_name"])
        save_parameter(self.elements, filename + 'elements',  self.config["rootdir"] + "\\" + self.config["folder_name"])


class read_mesh_from_csv(object):
    """
    Read mesh from csv file

    mode: 'csv' read from .csv file
          'pkl' read from .pkl cache
    """

    def __init__(self, name='Mesh_Cache', mode='pkl'):
        self.config = get_config()
        self.folder_name = self.config["folder_name"]
        if mode == 'csv':
            self.nodes = []
            self.elements = []
            with open(self.config["rootdir"] + "\\" + self.config["folder_name"] + '\\' + name + '_Node.csv', newline='') as datafile:
                csv_reader = csv.reader(datafile, delimiter=',')
                for line in csv_reader:
                    if line:
                        line = [float(x) for x in line]
                        self.nodes.append(line)
            with open(self.config["rootdir"] + "\\" + self.config["folder_name"] + '\\' + name + '_Element.csv', newline='') as datafile_2:
                csv_reader_2 = csv.reader(datafile_2, delimiter=',')
                for line in csv_reader_2:
                    if line:
                        line = [int(x) for x in line]
                        self.elements.append(line)
        elif mode == 'pkl':
            self.read_from_pkl()
        else:
            raise Exception('No such mesh reading mode.')
        # electrode information center, radius FOR FASTER PERFORMANCE PLEASE COPY FROM CONFIG
        self.electrode_centers = np.array(self.config["electrode_centers"])
        self.electrode_num = len(self.electrode_centers)
        self.electrode_radius = self.config["electrode_radius"]

    def read_from_pkl(self, filename='Mesh_'):
        self.nodes = read_parameter(filename + 'nodes', self.config["rootdir"] + "\\" + self.config["folder_name"])
        self.elements = read_parameter(filename + 'elements', self.config["rootdir"] + "\\" + self.config["folder_name"])

    def return_mesh(self):
        """
        Returns:
            mesh object, electrode number, electrode centers, electrode radius
        """
        element_num = len(self.elements)
        if self.config["mesh_unit"] == "mm":
            mesh_obj = {'element': np.array(self.elements), 'node': np.array(self.nodes) / 1000,
                        'perm': np.ones(element_num)}
        else:
            mesh_obj = {'element': np.array(self.elements), 'node': np.array(self.nodes), 'perm': np.ones(element_num)}
        return mesh_obj, self.electrode_num, self.electrode_centers, self.electrode_radius


def init_mesh(draw=False):
    """
    Run this after making new mesh,
    """
    config = get_config()
    filename = config["mesh_filename"]
    electrode_centers = config["electrode_centers"]
    electrode_radius = config["electrode_radius"]
    folder_name = config["folder_name"]
    optimize_node_num = config["optimize_node_num"]
    shuffle_element = config["shuffle_element"]
    is_SI = config["mesh_unit"] == "SI"
    read_mesh = ReadMesh(filename, electrode_centers, electrode_radius, folder_name, optimize_node_num, shuffle_element,
                         is_SI=is_SI)
    mesh_obj, electrode_num, electrode_centers, electrode_radius = read_mesh.return_mesh()
    if draw:
        draw_mesh(mesh_obj, electrode_num, electrode_centers, electrode_radius)
    return mesh_obj, electrode_num, electrode_centers, electrode_radius


def draw_mesh(mesh_obj, electrode_num, electrode_centers, electrode_radius):
    """
        Draw the mesh with electrode demonstration
    """

    plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    points = mesh_obj['node']
    tri = mesh_obj['element']
    perm = mesh_obj['perm']
    x, y = points[:, 0] * 0.7, points[:, 1] * 0.7
    fig, ax = plt.subplots(figsize=(4.25, 4.25))
    im = ax.tripcolor(x, y, tri, np.abs(perm), shading='flat', edgecolors='k', vmax=2, vmin=0)
    # fig.colorbar(im)
    for i, electrode_center in enumerate(electrode_centers):
        x = electrode_center[0] - electrode_radius
        y = electrode_center[1] - electrode_radius
        width = 2 * electrode_radius * 0.7
        ax.add_patch(
            patches.Rectangle(
                (x * 0.7, y * 0.7),  # (x,y)
                width,  # width
                width,  # height
                color='y'
            )
        )
        ax.annotate(str(i), (x * 0.7, y * 0.7))
    ax.set_aspect('equal')

    _, ax = plt.subplots(figsize=(20, 20))
    ax.plot(points[:, 0], points[:, 1], 'ro', markersize=5)
    for i in range(points.shape[0]):
        ax.text(points[i, 0], points[i, 1], str(i), fontsize=8)
    ax.grid('on')
    ax.set_aspect('equal')
    plt.show()


def demo_mesh():
    read_mesh = read_mesh_from_csv()
    mesh_obj, electrode_num, electrode_centers, electrode_radius = read_mesh.return_mesh()
    draw_mesh(mesh_obj, electrode_num, electrode_centers, electrode_radius)


if __name__ == "__main__":
    demo_mesh()
