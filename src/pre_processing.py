# %%
# Imports + Global variables
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
import pylab as pl
import xarray as xr


PATH_MESH = "../data/fesom.mesh.diag.nc"
PATHS_SSH = ["../data/ssh.fesom.2010.nc"]
#PATH_PILOT_SSH = "../data/pilot/raw/pilot_ssh.nc"
DEBUG_PLOT = True
DEBUG_DATA = True

# TODO: for every path in SSH, read the file and (delete the useless field)
# concatenate them, or read them with open_mfdataset()

# %%
# Read and print the mesh
data_mesh = xr.open_dataset(PATH_MESH, engine='netcdf4')
if DEBUG_DATA:
    print(data_mesh)

# %%
# Plot: entire world
model_lon = data_mesh.lon.values
model_lat = data_mesh.lat.values

if DEBUG_PLOT:
    step = 10
    plt.figure(figsize=(14, 6))
    plt.scatter(model_lon[::step], model_lat[::step], s=1)
    plt.xlabel('Longitude', size=10)
    plt.ylabel('Latitude', size=10)

# %%
# Deletion of useless fields 
vars_keys = data_mesh.data_vars
for key in vars_keys:
    if key != 'lat' and key != 'lon' and key != 'edges' and key != 'nodes':
        data_mesh = data_mesh.drop_vars(key)

data_mesh = data_mesh.drop_vars('nz')   # These 2 are coordinates, not variables
data_mesh = data_mesh.drop_vars('nz1')
if DEBUG_DATA:
    print(data_mesh)

# %%
# RoI & Plot: South Atlantic mask extraction (Eddy rich region)
left = 0
right = 1
bottom = -41
top = -40
region_mask = (model_lon > left) & (model_lon < right) & (model_lat < top) & (model_lat > bottom)

if DEBUG_PLOT:
    step = 1000
    plt.figure(figsize=(14, 6))
    plt.scatter(model_lon[region_mask][::step], model_lat[region_mask][::step], s=1)

# %%
# RoI: edges extraction
# Decrease by 1 since for compatibility with Fortran thhttps://github.com/massimilianofronza/unstructuredEarthe indexes start from 1 instead of 0
edge_0 = data_mesh.edges[0].values
edge_1 = data_mesh.edges[1].values
edge_0 -= 1
edge_1 -= 1

edges_subset = []
for i in range(len(edge_0)):
    if region_mask[edge_0[i]] & region_mask[edge_1[i]]:
        edges_subset.append([edge_0[i], edge_1[i]])

edges_subset = np.array(edges_subset, dtype="int32")
data_mesh = data_mesh.drop_vars('edges')
data_mesh['edges'] = (('edges_subset', 'n2'), edges_subset)
if DEBUG_DATA:
    print(data_mesh)

# %%
# RoI: nodes extraction
nodes_subset = []
for i in range(len(region_mask)):
    if region_mask[i]:
        nodes_subset.append(i)

nodes_subset = np.array(nodes_subset, dtype="int32")
data_mesh['nodes'] =(('nodes_subset'), nodes_subset)
if DEBUG_DATA:
    print(data_mesh)

# %%
# Plot: final RoI
model_lon_roi = data_mesh.lon[data_mesh.nodes].values
model_lat_roi = data_mesh.lat[data_mesh.nodes].values

if DEBUG_PLOT:
    step = 1
    plt.figure(figsize=(14, 6))
    plt.scatter(model_lon_roi[::step], model_lat_roi[::step], s=1)
    plt.xlabel('Longitude', size=10)
    plt.ylabel('Latitude', size=10)

# %%
# Plot: just the connections of the RoI
edge_0 = data_mesh.edges[:, 0].values
edge_1 = data_mesh.edges[:, 1].values
x_0 = []
y_0 = []
x_1 = []
y_1 = []
for i in range(edge_0.size):
    if region_mask[edge_0[i]] & region_mask[edge_1[i]]:
        # First the connection starting points
        x_0.append(float(model_lon[edge_0[i]]))
        y_0.append(float(model_lat[edge_0[i]]))
        # Then the connection end points
        x_1.append(float(model_lon[edge_1[i]]))
        y_1.append(float(model_lat[edge_1[i]]))

# Organizing the start and ending connection points for visualization
lines = []
for i in range(len(x_0)):
    lines.append([(x_0[i], y_0[i]), (x_1[i], y_1[i])])

# Color mask to highlight the last line
c = [(0, 1, 0, 1)]*(len(x_0)-1)
#c.append((1, 0, 0, 1))
c.append((0, 1, 0, 1))

lc = mc.LineCollection(lines, colors=c, linewidths=1)
fig, ax = pl.subplots()
ax.add_collection(lc)
#for i in range(len(x_0)):
#    plt.annotate(f"start_{i}", (x_0[i], y_0[i]))
#    plt.annotate(f"end_{i}", (x_1[i], y_1[i]))
ax.margins(0.1)
plt.show()

# %%
