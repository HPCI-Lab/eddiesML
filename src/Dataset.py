import glob
import numpy as np
import os
import random
import torch
import torch_geometric.utils as tg_utils
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import xarray as xr

class EddyDataset(Dataset):
    # root: Where the dataset should be stored and divided into processed/ and raw/
    def __init__(self, root, mesh_path, dataset_size, split, proportions, random_seed, transform=None, pre_transform=None, pre_filter=None):
        
        self.mesh_path = mesh_path
        self.split = split
        
        # Call of process() within
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # These are useless(TODO for now?)è
        graph_names = self.processed_file_names
        if 'pre_filter.pt' in graph_names:
            os.remove(self.processed_dir + '/pre_filter.pt')
        if 'pre_transform.pt' in graph_names:
            os.remove(self.processed_dir + '/pre_transform.pt')
        
        TRAIN_PROP, VAL_PROP, TEST_PROP = proportions[0], proportions[1], proportions[2]
        
        if dataset_size > len(self.processed_file_names):
            raise ValueError(f"Parameter 'dataset_size' with value {dataset_size} is bigger than the available samples, which are {len(self.processed_file_names)}")
        
        self.n_train = round(dataset_size*TRAIN_PROP/100)
        self.n_val = round(dataset_size*VAL_PROP/100)
        self.n_test = dataset_size-self.n_train-self.n_val
        
        # The order is going to be the same for train, val and test due to the shared random_seed
        self.permutations = [_ for _ in range(len(self.processed_file_names))]
        random.seed(random_seed)
        random.shuffle(self.permutations)
        
        if split == 'train':
            self.permutations = self.permutations[:self.n_train]
        elif split == 'val':
            self.permutations = self.permutations[self.n_train:(self.n_train+self.n_val)]
        elif split == 'test':
            self.permutations = self.permutations[(self.n_train+self.n_val):(self.n_train+self.n_val+self.n_test)]
        
        # Just print this once
        if split == 'train':
            print("    Shape of node feature matrix:", np.shape(self[0].x))
            print("    Shape of graph connectivity in COO format:", np.shape(self[0].edge_index))
            print("    Shape of labels:", np.shape(self[0].y))
    
    @property
    # If you directly return the names of the files, the '.' will be in /data/bsc/raw/
    # If you return os.listdir, the '.' will be where "Dataset.py" is
    def raw_file_names(self):
        return os.listdir(self.root + '/raw')

    @property
    def processed_file_names(self):
        return os.listdir(self.root + '/processed')
    
    # Converts files in /raw into graphs in processed/
    # This function is triggered as soon as the PilotDataset is instantiated
    def process(self):
        
        # Get the adjacency info(common for all our graphs)
        edge_index = self._get_adjacency_info()
        
        node_feats = None
        labels = None
        
        for raw_path in self.raw_paths:
            file_name = raw_path.split('/')[-1]
            year = file_name.split('_')[2]
            month = file_name.split('_')[3]
            day = file_name.split('_')[4].split('.')[0]
            
            #print(f'    Year {year}, Month {month}, Day {day}...')
            
            raw_data = xr.open_dataset(raw_path)

            # Get node features
            node_feats = self._get_node_features(raw_data)
            
            # Get labels info
            labels = self._get_labels(raw_data)

            # Create the Data object
            data = Data(
                x=node_feats,                       # node features
                edge_index=edge_index,              # edge connectivity
                y=labels,                           # labels for classification
            )

            torch.save(data, os.path.join(self.processed_dir, f'year_{year}_month_{month}_day_{day}.pt'))


    # Return the SSH information with shape=[num_nodes, num_node_features]
    def _get_node_features(self, data):
        all_nodes_feats = []
        nodes_feats = data.ssh.values
        all_nodes_feats.append(nodes_feats)
        all_nodes_feats = np.asarray(all_nodes_feats)
        all_nodes_feats = all_nodes_feats.T
        return torch.tensor(all_nodes_feats, dtype=torch.float)

    # Return the graph edges in COO format with shape=[2, num_edges]
    # TODO I created a edges_local variable that restricts the node indexes between [0, num_nodes], so now we should be fine
    def _get_adjacency_info(self):
        mesh = xr.open_dataset(self.mesh_path)
        edges_coo = torch.tensor(mesh.edges_local.values, dtype=torch.long)
        edges_coo = tg_utils.to_undirected(edges_coo)
        return edges_coo
    
    # Return the segmentation mask in the form of labels for graph nodes
    def _get_labels(self, data):
        labels = data.seg_mask.values
        return torch.tensor(labels, dtype=torch.long)
    
    # Download the raw data into raw/, or the folder specified in self.raw_dir
    def download(self):
        pass

    # Returns the number of examples in the dataset
    def len(self):
        return len(self.permutations)

    # Implements the logic to load a single graph - TODO with a lot of data this slows the process quite a lot
    def get(self, idx):
        files = [self.processed_file_names[p] for p in self.permutations]
        data = files[idx]
        data = torch.load(os.path.join(self.processed_dir, data))
        return data

    # Gets files per year, month, and/or day - TODO: NOT USED ANYWHERE
    def get_by_time(self, year=None, month=None, day=None):
        if year == None:
            year = '*'
        if month == None:
            month = '*'
        if day == None:
            day = '*'
        file_names = os.path.join(self.processed_dir, f"year_{year}_month_{month}_day_{day}.pt")
        
        files = []
        for file in glob.glob(file_names):
            files.append(torch.load(file))
        return files
