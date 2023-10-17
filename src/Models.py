import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphUNet
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dropout_edge


# Graph U-Net class
class GUNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, final_act):
        super().__init__()

        self.act_middle = torch.nn.functional.relu
        self.act_final = final_act
        
        pool_ratios = [2000 / num_nodes, 0.5]
        self.unet = GraphUNet(in_channels, hidden_channels, out_channels,
                              depth=3, pool_ratios=pool_ratios, act=self.act_middle)
        
        self._log_network()
        
    def forward(self, data):
        edge_index, _ = dropout_edge(data.edge_index, p=0.2,
                                     force_undirected=True,
                                     training=self.training)
        x = F.dropout(data.x, p=0.92, training=self.training)
        x = self.unet(x, edge_index)

        act = None
        if self.act_final.__name__ == "softmax":
            act = self.act_final(x, dim=1)
        elif self.act_final.__name__ == "sigmoid":
            act = self.act_final(x)
        return act

    def _log_network(self):
        middle = self.act_middle.__name__
        final = self.act_final.__name__
        print(f"GUNet instantiated!\n\tMiddle act: {middle}\n\tFinal act: {final}")
    
# A GCN model created for comparison purposes with the Graph U-Net model
class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, num_layers):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        
        # Add the input layer
        self.conv_layers.append(GCNConv(in_channels, hidden_channels))
        
        # Add the hidden layers
        for _ in range(num_layers - 2):  # -2 because we already added input and will add output later
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
        
        # Add the output layer
        self.conv_layers.append(GCNConv(hidden_channels, out_channels))
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.conv_layers:
            x = F.relu(layer(x, edge_index))
        return F.log_softmax(x, dim=1)

    def _log_network(self):
        middle = nn.Relu()
        final = nn.Softmax()
        print(f"GCN instantiated!\n\tMiddle act: {middle}")#\n\tFinal act: {final}")
        
# Define the GraphSAGE model class
class GraphSAGEModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, normalize=True):
        super(GraphSAGEModel, self).__init__()

        self.num_layers = num_layers

        # List to store the GraphSAGE convolutional layers
        self.conv_layers = nn.ModuleList()

        # Input layer
        self.conv_layers.append(SAGEConv(num_features, hidden_dim))

        # Apply He initialization to the underlying linear transformations
        for i in range(len(self.conv_layers)):
            nn.init.kaiming_uniform_(self.conv_layers[i].lin_l.weight)
            nn.init.kaiming_uniform_(self.conv_layers[i].lin_r.weight)

        # Intermediate layers
        for _ in range(1, num_layers - 1):
            self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim))

        # Output layer
        self.conv_layers.append(SAGEConv(hidden_dim, num_classes))

        self.normalize = normalize

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)

            if i < self.num_layers - 1:
                x = F.relu(x)

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)  # L2 normalization

        return F.log_softmax(x, dim=1)
    
    ## Possible improvements points
    # Number of layer might significantly impact its performance
    # Different hidden dims (e.g., 64, 128, 256)
    # Include or exclude L2 normalization - try
    # Number of Aggregation Neighbors (num_samples) - accuracy vs performance
    #Batch size, epoch, optimizer loss