# FESOM2 unstructured pipeline
Python scripts for the processing of unstrutured FESOM2 grids for the identification of oceanic eddies.

## List of files in src:
 * pre_processing.ipynb           Creation of subset mesh and interpolation of SSH and seg_mask to unstructured space
 * pre_processing_demo.ipynb      Graphical demonstration notebook for the pre-processing
 * pipeline_demo.ipynb            First instance of the DL pipeline's main body
 * Dataset.py                     PyTorch Dataset class to create the set of graphs that will compose train, validation and test sets
