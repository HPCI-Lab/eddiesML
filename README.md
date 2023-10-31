# eddiesGNN

### Description

This repository contains the modules of a deep learning pipeline to pre-process FESOM2 data and train a Graph Neural Network on said unstructured meshes.
The network can then be used to detect oceanic mesoscale eddies directly on the unstructured grid, without the need for prior interpolation to regular matrices.

### How to Setup

For this process, it's preferable to have an installation of Anaconda/Miniconda. Having a recent enough version of Python is essential for using features like the GPU computation.

```
$ conda create -n env_eddiesGNN python=3.11
$ conda activate env_eddiesGNN
$ pip install -r requirements.txt
```

If you want to also create the link between your environment and your JupyterHub, use this:

```
$ python -m ipykernel install --user --name env_eddiesGNN --display-name="<env_name>"
```

### Components of this Repository

##### pre_processing.ipynb

Creates the subset unstructured mesh (if not already existing) from the global one.<br>
Interpolates the SSH and its segmentation mask from the regular matrix space to the unstructured subset mesh, using the K-Nearest Neighbors algorithm.<br>
Applies some corrections in the data before writing the final output.<br>
Requires a JupyterHub server to be run.

##### pre_processing_demo.ipynb

Demonstrates the pre-processing phase by repeating the same steps of pre_processing.ipynb, but with more explanations, plots, and without writing the final results on the file system.<br>
Requires a JupyterHub server to be run.

##### pipeline.py

The core body of the training/testing phase. It uses the PyTorch Geometric framework and the code in Dataset.py, Loss.py, and Model.py.<br>
Creates the graph dataset, splits it into train-validation-test, tests the hyperparameters, creates the DataLoaders, instantiates the Graph U-Net, the Optimizer, the Dice Loss, performs the actual training, and outputs some comparison plots.<br>
Requires a Slurm execution script to be scheduled, like pipeline.sh.<br>
GPU support is native with the code.

##### pipeline_demo.ipynb

Performs the training/testing phase in a notebook by repeating the same steps of pipeline.py, but with more explanations.<br>
Requires a JupyterHub server to be run.<br>
GPU support is native with the code.
