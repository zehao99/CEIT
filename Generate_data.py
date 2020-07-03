import numpy as np
import csv
from MyEIT.efem import EFEM
from MyEIT.readmesh import read_mesh_from_csv_mm

def write_data(capa, potential_readings):
    with open('learning_data/capacitance.csv', "a", newline= '') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(capa)
    
    with open('learning_data/potential_data.csv', "a", newline= '') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(potential_readings)


if __name__ == "__main__":

    """ 0. construct mesh """
    read_mesh = read_mesh_from_csv_mm()
    mesh_obj, electrode_num, electrode_centers, radius = read_mesh.return_mesh()
    # extract node, element, alpha
    points = mesh_obj['node']
    elem = mesh_obj['element']
    x, y = points[:, 0], points[:, 1]

    """ 1. problem setup """
    fwd = EFEM(mesh_obj,electrode_num ,electrode_centers, radius, perm = 1/200000)
    for i in range(100000):
        num_of_shape = np.random.rand() * 3
        if num_of_shape < 1:
            x1 = (-40 + 80 * np.random.rand()) / 1000
            y1 = (-40 + 80 * np.random.rand()) / 1000
            radius1 = (20 * np.random.rand() + 1.5) / 1000
            value1 = 10 ** (np.random.rand() * 4 - 10)
            if np.random.rand() < 0.5:
                shape1 = "square"
            else:
                shape1 = "circle"
            fwd.change_add_capa_geometry([x1,y1], radius1, value1, shape1)
            electrode_potential = []
            for j in range(16):
                node_u, elem_u, electrode_potential_single = fwd.calculation(j)
                electrode_potential += list(np.abs(electrode_potential_single))
            write_data(fwd.elem_capacitance,electrode_potential)
            fwd.change_add_capa_geometry([x1,y1], radius1, -value1, shape1)
        elif num_of_shape < 2:
            x1 = (-40 + 80 * np.random.rand()) / 1000
            y1 = (-40 + 80 * np.random.rand()) / 1000
            radius1 = (20 * np.random.rand() + 1.5) / 1000
            value1 = 10 ** (np.random.rand() * 4 - 10)
            if np.random.rand() < 0.5:
                shape1 = "square"
            else:
                shape1 = "circle"
            x2 = (-40 + 80 * np.random.rand()) / 1000
            y2 = (-40 + 80 * np.random.rand()) / 1000
            radius2 = (20 * np.random.rand() + 1.5) / 1000
            value2 = 10 ** (np.random.rand() * 4 - 10)
            if np.random.rand() < 0.5:
                shape2 = "square"
            else:
                shape2 = "circle"
            fwd.change_add_capa_geometry([x1,y1], radius1, value1, shape1)
            fwd.change_add_capa_geometry([x2,y2], radius2, value2, shape2)
            electrode_potential = []
            for j in range(16):
                node_u, elem_u, electrode_potential_single = fwd.calculation(j)
                electrode_potential += list(np.abs(electrode_potential_single))
            write_data(fwd.elem_capacitance,electrode_potential)
            fwd.change_add_capa_geometry([x1,y1], radius1, -value1, shape1)
            fwd.change_add_capa_geometry([x2,y2], radius2, -value2, shape2)
        else:
            x1 = (-40 + 80 * np.random.rand()) / 1000
            y1 = (-40 + 80 * np.random.rand()) / 1000
            radius1 = (20 * np.random.rand() + 1.5) / 1000
            value1 = 10 ** (np.random.rand() * 4 - 10)
            if np.random.rand() < 0.5:
                shape1 = "square"
            else:
                shape1 = "circle"
            x2 = (-40 + 80 * np.random.rand()) / 1000
            y2 = (-40 + 80 * np.random.rand()) / 1000
            radius2 = (20 * np.random.rand() + 1.5) / 1000
            value2 = 10 ** (np.random.rand() * 4 - 10)
            if np.random.rand() < 0.5:
                shape2 = "square"
            else:
                shape2 = "circle"
            x3 = (-40 + 80 * np.random.rand()) / 1000
            y3 = (-40 + 80 * np.random.rand()) / 1000
            radius3 = (20 * np.random.rand() + 1.5) / 1000
            value3 = 10 ** (np.random.rand() * 4 - 10)
            if np.random.rand() < 0.5:
                shape3 = "square"
            else:
                shape3 = "circle"
            fwd.change_add_capa_geometry([x1,y1], radius1, value1, shape1)
            fwd.change_add_capa_geometry([x2,y2], radius2, value2, shape2)
            fwd.change_add_capa_geometry([x3,y3], radius3, value3, shape3)
            electrode_potential = []
            for j in range(16):
                node_u, elem_u, electrode_potential_single = fwd.calculation(j)
                electrode_potential += list(np.abs(electrode_potential_single))
            write_data(fwd.elem_capacitance,electrode_potential)
            fwd.change_add_capa_geometry([x1,y1], radius1, -value1, shape1)
            fwd.change_add_capa_geometry([x2,y2], radius2, -value2, shape2)
            fwd.change_add_capa_geometry([x3,y3], radius3, -value3, shape3)
        if i % 10 == 0:
            print("iteration to: " , i)