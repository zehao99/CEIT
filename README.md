# CEIT

Package for Electric Impedance Tomography on detecting Capacitance Density

## Overview

This package is specifically designed to solve the tomographic problem concerned with detecting proximity map by a planar conductive sensor.

For more information, please check my paper.

The `efem.py` module is written only for this problem, other modules can be reused in any other EIT application.

CEIT provides the ability to generate solver for realtime reconstruction.
Given the meshes and electrode positions, CEIT can generate Inverse model for any planar sensor design.

## Requirements

See `requirements.txt`, one thing to mention is that to accelerate the calculation process, we used GPU acceleration for matrix multiplication.
So if you don't have a beefy GPU, then please set the device option in `config.json` to `"cpu"` and do the following things:
> 1. You sure cannot install `cupy` without CUDA, remove it from the `requirements.txt`.
> 2. comment out content inside function `calculate_FEM_equation()` at the end of file `./MyEIT/efem.py`.
> 3. Add a line `pass` to the function.
> 4. comment out `import cupy as cp` in `./MyEIT/efem.py`.

## Configure the calculation

You should configure the `config.json` file before using this package.

A `.fem` file is needed for initializing the whole process. You can get one by using CAD software.

Also, you have to decide your electrode center positions, and your radius of the electrode.
Inside this package, the electrode is square shaped for which the radius means **half width** of the square.

For Examples see `config.json` file.

| Parameter Name | Type | Description |
| :----: | :----: |:----:|
| `"mesh_filename"` | `String` | File name for your mesh file |
| `"folder_name"` | `String` | Specify the folder you push your mesh file and all the cache files.|
| `"optimize_node_num"`| `Boolean` | Whether shuffle node number at initializing mesh |
| `"shuffle_element"` | `Boolean` | Whether shuffle elemets at initializing mesh |
| `"electrode_centers"` | `Array` | Center of electrodes on perimeter THE UNIT IS **mm** |
| `"electrode_radius"`| `Number` | In this package electrodes are treated as square shaped, this parameter is half of its side length.
| `"capacitance_change_for_JAC"` |`Number`| Capacitance change on every single element when calculating the Jacobian matrix.|
| `"detection_bound"`| `Number` | Specify the detection boundary size please keep its unit identical to the `"unit"` property|
| `"calc_from"`| `Number` | Set starting electrode for Jacobian calculation, for multiple instances compute usage.
| `"calc_end"` | `Number` | Set ending electrode for Jacobian calculation, for multiple instances compute usage.
| `"regularization_coeff"` | `Number` | This parameter is used in regularization equation of reconstruction, **you will have to optimize it**.
| `"device"` |  `String` | Calculation device, only `"cpu"` or `"gpu"` is accepted, if you choose `"cpu"` please follow the instructions in the previous paragraph.|
| `"unit"` | `String` | Unit for the input above. Only `"mm"` or `"SI"` is accepted, they will all be transferred into SI unit, please keep the units inside mesh file and config file the same.|
| `"reconstruction_mode"` |`String`| DEPRECATED ITEM keep this to `"n"`|
| `"overall_origin_capacitance"` |`Number`| DEPRECATED ITEM keep this to `0`|

## Quick Start

Here is a sample for simple forward calculation using this package.

```python
import matplotlib.pyplot as plt
from MyEIT.efem import EFEM
from MyEIT.readmesh import read_mesh_from_csv

""" Read mesh from csv files(after initialization) """

read_mesh = read_mesh_from_csv()
mesh_obj, electrode_num, electrode_centers, radius = read_mesh.return_mesh()

""" problem setup """

fwd = EFEM(mesh_obj, electrode_num, electrode_centers, radius, perm=1 / 200000)
fwd.change_add_capa_geometry([-0.02, -0.01], 0.01, 1e-7, 'square')
node_u, elem_u, electrode_potential = fwd.calculation(2)

print(electrode_potential)

""" Visualization """

fig, ax = plt.subplots(figsize=(3.2, 3.2))
im = fwd.plot_potential_map(ax)
plt.colorbar(im)
plt.show()
```

## Read Mesh Class

All the mesh initializer is put in `MyEIT.readmesh`.

Now the reader only work with `.fem` files generated by Altair HyperMesh

### 1. Initialize a new mesh
First you should finish configuring your `config.json` file according to the previous paragraph.

Then on the first time running of a new mesh, call `init_mesh()` function to initialize mesh.

When initializing, the class will automatically clear out duplicated mesh and you can decide whether it 
should shuffle the mesh number or not.

After initializing, in the folder you specified before, the method would generate a `Mesh_Cache_Node.csv` 
file and a `Mesh_Cache_Element.csv` file.

```python
from MyEIT.readmesh import init_mesh

init_mesh(draw=True)
```

### 2. Read from generated mesh cache

After initializing the mesh, you can quickly read from the cache file.

Class `read_mesh_from_csv` provides function to read mesh from csv file.
The default calculation unit inside CEIT is **SI** units, if your mesh is in **mm** unit, please set in `config.json` file.

**You need to call `return_mesh()` method to get the mesh object and electrode information.**

```python
from MyEIT.readmesh import read_mesh_from_csv

read_mesh = read_mesh_from_csv()
mesh_obj, electrode_num, electrode_centers, electrode_radius = read_mesh.return_mesh()
```

## Forward Calculator

The forward calculator is used Finite Element Method to calculate potential distribution on the surface.

First instantiate the class
```python
from MyEIT.efem import EFEM

fwd_model = EFEM(mesh_obj, electrode_num, electrode_centers, electrode_radius)
```

The initializer will automatically prepare the object for calculation, now you have a fully functioning forward solver.

There are several functions provided by this object you can call to change capacitance value and do calculation.

|function name|Description|
|:----:|:----:|
|`EFEM.calculation(electrode_input)`|Forward calculation on given input electrode selection **You have to call this to do the calculation**|
|`EFEM.plot_potential_map(ax)`|Plot the current forward result, all `0` before calling `calculation()`|
|`EFEM.plot_current_capacitance(ax)`|Plot the given input condition|
|`EFEM.change_capacitance_elementwise(element_list, capacitance_list)`|Change capacitance density on selected elements|
|`EFEM.change_capacitance_geometry(center, radius, value, shape)`|**Assign** capacitance density on elements inside a certain geometry (square or circle) to the given value|
|`EFEM.change_add_capa_geometry(center, radius, value, shape)`|**Add** the given capacitance density on elements inside a certain geometry|
|`EFEM.change_conductivity(element_list, resistance_list)`|Change conductivity on certain elements|
|`EFEM.reset_capacitance(overall_capa)`|Set capacitance density on all elements to `overall_capa`|

## Jacobian Constructor

## Realtime Solver
