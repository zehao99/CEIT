import matplotlib.pyplot as plt
from MyEIT.efem import EFEM
from MyEIT.readmesh import read_mesh_from_csv_SI, init_mesh

""" Read mesh from csv files(after initialization) """
read_mesh = read_mesh_from_csv_SI()
mesh_obj, electrode_num, electrode_centers, radius = read_mesh.return_mesh()
# extract node, element, alpha
""" problem setup """
fwd = EFEM(mesh_obj, electrode_num, electrode_centers, radius, perm=1 / 200000)
fwd.change_add_capa_geometry([-0.02, -0.01], 0.01, 1e-7, 'square')
node_u, elem_u, electrode_potential = fwd.calculation(2)

print(electrode_potential)
# Visualization
fig, ax = plt.subplots(figsize=(3.2, 3.2))
im = fwd.plot_potential_map(ax)
plt.colorbar(im)
plt.show()
