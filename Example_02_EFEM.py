import matplotlib.pyplot as plt
from MyEIT.EFEM import EFEM
from MyEIT.readmesh import read_mesh_from_csv
from MyEIT.models.mesh import MeshObj

# problem setup
fwd = EFEM()

obj_x = -20  # object x position
obj_y = -10  # object y position
obj_r = 1    # object radius
c_val = 1e-2    # object variable density val
obj_shape = "square"    # object shape

fwd.change_add_variable_geometry(
    [obj_x / 1000, obj_y / 1000], obj_r / 1000, c_val, obj_shape)
node_u, elem_u, electrode_potential = fwd.calculation(2)

print(electrode_potential)

# Visualization
fig, ax = plt.subplots(figsize=(3.2, 3.2))
im = fwd.plot_potential_map(ax)
plt.colorbar(im)
plt.show()
