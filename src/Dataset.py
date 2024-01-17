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
        self.scaler_feats = None
        
        # Call of process() within
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # These are useless(TODO for now?)
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
        
        self.permutations = [self.processed_file_names[p] for p in self.permutations]
        
        # Just print this once
        if split == 'train':
            print("    Shape of node feature matrix:", np.shape(self[0].x))
            print("    Shape of graph connectivity in COO format:", np.shape(self[0].edge_index))
            print("    Shape of edge weights:", np.shape(self[0].edge_attr))
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
        
        # Get the adjacency info(common for all the graphs)
        edge_index = self._get_adjacency_info()
        
        # Get the edge features(common for all the graphs)
        edge_attr = self._get_edge_features(edge_index)
        
        node_feats = None
        labels = None
        
        for raw_path in self.raw_paths:
            file_name = raw_path.split('/')[-1]
            year = file_name.split('_')[1]
            month = file_name.split('_')[2]
            day = file_name.split('_')[3].split('.')[0]
            
            #print(f'    Year {year}, Month {month}, Day {day}...')
            
            raw_data = xr.open_dataset(raw_path)

            # Get node features
            node_feats = self._get_node_features(raw_data)
            
            # Get labels info
            labels = self._get_labels(raw_data)

            # Create the Data object
            data = Data(
                x=node_feats,                  # node features
                edge_index=edge_index,         # edge connectivity
                #edge_attr=edge_attr,           # edge attributes
                y=labels,                      # labels for classification
            )

            torch.save(data, os.path.join(self.processed_dir, f'year_{year}_month_{month}_day_{day}.pt'))

    
    # Return the Haversine distance between 2 geographic points
    def _haversine_dist(self, lon1, lat1, lon2, lat2):
        # Radius of the Earth in meters
        R = 6371000

        # Convert lon and lat from degrees to radians
        lon1, lat1, lon2, lat2 = np.radians([lon1, lat1, lon2, lat2])

        # Haversine formula
        dlon = lon2-lon1
        dlat = lat2-lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        dist = R*c
        return dist

    # Return the edges lengths in meters with shape=[num_edges, num_edge_features]
    def _get_edge_features(self, edge_index):
        all_edges_feats = []
        distances = []
        
        mesh = xr.open_dataset(self.mesh_path)
        edge_index = np.array(edge_index)

        lon = mesh.lon.values
        lat = mesh.lat.values
        nodes = mesh.nodes.values

        edges_lon = lon[nodes[edge_index]]
        edges_lat = lat[nodes[edge_index]]
        
        for i in range(np.shape(edges_lon)[1]):
            lon1 = edges_lon[0, i]
            lat1 = edges_lat[0, i]
            lon2 = edges_lon[1, i]
            lat2 = edges_lat[1, i]
            dist = self._haversine_dist(lon1, lat1, lon2, lat2)
            distances.append(dist)
        
        all_edges_feats.append(distances)
        all_edges_feats = np.asarray(all_edges_feats)
        all_edges_feats = all_edges_feats.T
        
        return torch.tensor(all_edges_feats, dtype=torch.float)
    
    # Return the SSH information with shape=[num_nodes, num_node_features]
    def _get_node_features(self, data):
        all_nodes_feats = []

        # Append all available variables except for the label information
        for key in data.data_vars:
            if key != 'seg_mask':
                all_nodes_feats.append(data.data_vars[key].values)

        all_nodes_feats = np.asarray(all_nodes_feats)
        all_nodes_feats = all_nodes_feats.T
        return torch.tensor(all_nodes_feats, dtype=torch.float)

    # Return the graph edges in COO format with shape=[2, num_edges]
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

    # Loads a single graph, scaled if a scaler is present
    def get(self, idx):
        data = self.permutations[idx]
        data = torch.load(os.path.join(self.processed_dir, data))
        if self.scaler_feats != None:
            data.x = torch.tensor(self.scaler_feats.transform(data.x), dtype=torch.float)
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
