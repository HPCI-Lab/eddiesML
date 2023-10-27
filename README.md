# eddiesGNN

### Description

The eddiesGNN repository contains the modules of a deep learning pipeline that can pre-process and train a Graph Neural Network on FESOM2 unstructured meshes.
The network can then be used to detect oceanic mesoscale eddies directly on the unstructured grid, without the need for prior interpolation to regular matrices.

### Requirements

In order to run this deep learning pipeline, make sure you have the following packages(all available via Pip or Conda) properly installed:

1. ipykernel
2. matplotlib
3. netcdf4
4. pyyaml
5. scikit-learn
6. tabulate
7. torch
8. torch-scatter
9. torch-sparse
10. torch-geometric
11. xarray

**Note**:
The libraries above are defined also in the requirements.txt file.

### How to Setup

For this process, you need an installation of Anaconda. Having a recent enough version of Python is essential for using features like the GPU computation.

```
$ conda create -n env_eddiesGNN python=3.11
$ conda activate env_eddiesGNN
$ pip install -r requirements.txt
```

If you want to also create the link between your environment and your JupyterHub, use this:

```
$ python -m ipykernel install --user --name env_eddiesGNN --display-name="<env_name>"
```

### List of Components

 * pre_processing.ipynb
   * Creation of subset mesh and interpolation of SSH and seg_mask to unstructured space
 * pre_processing_demo.ipynb
   * Graphical demonstration notebook for the pre-processing
 * pipeline_demo.ipynb
   * First instance of the DL pipeline's main body
 * Dataset.py
   * PyTorch Dataset class to create the set of graphs that will compose train, validation and test sets

