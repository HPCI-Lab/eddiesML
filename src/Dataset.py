import numpy as np
import os
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import xarray as xr

class PilotDataset(Dataset):

    # root: Where the dataset should be stored and divided into processed/ and raw/
    def __init__(self, root, label_type, transform=None, pre_transform=None, pre_filter=None):
        self.label_type = label_type # TODO remove this
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    # If you directly return the names of the files, the '.' will be in /data/bsc/raw/
    # If you return os.listdir, the '.' will be where "Dataset.py" is
    def raw_file_names(self):
        return os.listdir(self.root + '/raw')

    @property
    # Return a list of graph files in the processed/ folder.
    # If these files:
    #   don't exist: process() will start and create them
    #   exist: process() will be skipped
    def processed_file_names(self):
        return os.listdir(self.root + '/processed')
    
    # Read the raw data and convert it into graph representations that are going to be saved into the processed/ folder.
    # This function is triggered as soon as the PilotDataset is instantiated
    def process(self):
        
        # Conserving adjacency info to avoid computing it every time
        edge_index = None

        for raw_path in self.raw_paths:

            # TODO for each file I was extracting year and cyclone number. Here we'll have something else
            year = raw_path.split('_')[2]
            cyclone = raw_path.split('_')[4].split('.')[0]
            #print(f'    Year {year}, Patch number {cyclone}...')
            raw_data = xr.open_dataset(raw_path)

            # Get node features
            node_feats = self._get_node_features(raw_data)

            # Get edge features - TODO do we need to assign here the physical distance between nodes?
            #edge_feats = self._get_edge_features(raw_data)

            # Get adjacency info
            if edge_index==None:
                edge_index = self._get_adjacency_info(raw_data)

            # Get labels info
            labels = self._get_labels(raw_data)

            # Create the Data object
            data = Data(
                x=node_feats,                       # node features
                edge_index=edge_index,              # edge connectivity
                y=labels,                           # labels for classification
            )

            #print(os.path.join(self.processed_dir, f'year_{year}_cyclone_{cyclone}.pt'))
            # TODO customize this to fit the type of eddy information we're going to store
            torch.save(data, os.path.join(self.processed_dir, f'year_{year}_cyclone_{cyclone}.pt'))

        print("    Shape of node feature matrix:", np.shape(node_feats))
        print("    Shape of graph connectivity in COO format:", np.shape(edge_index))
        print("    Shape of labels:", np.shape(labels))


    # Return a matrix with shape=[num_nodes, num_node_features]
    # Features here are the SSH information
    def _get_node_features(self, data):

        all_nodes_feats =[]

        # TODO for every node, append its SSH information to the above list

        all_nodes_feats = np.asarray(all_nodes_feats)
        return torch.tensor(all_nodes_feats, dtype=torch.float)


    # Retrieve the edge features
    def _get_edge_features(self, data):
        all_edge_feats = []
        all_edge_feats = np.asarray(all_edge_feats)
        return all_edge_feats


    # Return the graph connetivity in COO format with shape=[2, num_edges]
    def _get_adjacency_info(self, data):
        
        coo_links = [[], []]

        # TODO for every edge, append start and end node in coo_links[0] and coo_links[1]

        return torch.tensor(coo_links, dtype=torch.long)


    # Download the raw data into raw/, or the folder specified in self.raw_dir
    def download(self):
        pass

    # Returns the number of examples in the dataset
    def len(self):
        return len(self.processed_file_names)
    
    # Implements the logic to load a single graph - TODO we'll have to redefine this
    def get(self, year, cyclone):
        data = torch.load(os.path.join(self.processed_dir, f'year_{year}_cyclone_{cyclone}.pt'))
        return data
