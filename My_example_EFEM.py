import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.patches as patches
from MyEIT.efem import EFEM
from MyEIT.readmesh import read_mesh_from_csv_mm

if __name__ == "__main__":
    
    del matplotlib.font_manager.weight_dict['roman']
    matplotlib.font_manager._rebuild()

    """ 0. construct mesh """
    read_mesh = read_mesh_from_csv_mm()
    mesh_obj, electrode_num, electrode_centers, radius = read_mesh.return_mesh()
    # extract node, element, alpha
    points = mesh_obj['node']
    elem = mesh_obj['element']
    x, y = points[:, 0], points[:, 1]
    """ 1. problem setup """
    fwd = EFEM(mesh_obj,electrode_num ,electrode_centers, radius, perm = 1/200000)
    #fwd.elem_capacitance = fwd.elem_capacitance + 1e-10
    #fwd.reset_capacitance(1e-13)
    #_, _, electrode_potential_orig = fwd.calculation(0)
    fwd.change_add_capa_geometry([-0.02,-0.01],0.01, 1e-7, 'square')
    #fwd.change_capacitance_elementwise([2500],[1e-5])
    node_u, elem_u, electrode_potential = fwd.calculation(2)

    print(electrode_potential)#-electrode_potential_orig)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)

    fig, ax = plt.subplots(figsize=(2, 2))
    triang = tri.Triangulation(x, y, elem)
    im = ax.tripcolor(triang, np.real(elem_u), shading='flat',cmap='plasma')
    fig.colorbar(im)
    
    
    ax.set_aspect('equal')
    #plt.savefig('GT2Elec2.png')

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    im = ax.tripcolor(x, y, elem, np.abs(fwd.elem_capacitance), shading='flat', cmap='plasma')
    #fig.colorbar(im)
    ax.set_aspect('equal')
    for i, electrode_center in enumerate(electrode_centers):
            x0 = electrode_center[0] - radius
            y0 = electrode_center[1] - radius
            width = 2 * radius
            ax.add_patch(
                patches.Rectangle(
                (x0, y0),   # (x,y)
                width,          # width
                width,          # height
                color='k'
            )
            )
    ax.add_patch(
        patches.Rectangle(
            (-50, -50),   # (x,y)
            100,          # width
            100,          # height
            color='k',
            fill=False
        )
    )
    #plt.savefig('GT2.png')

    plt.show()

