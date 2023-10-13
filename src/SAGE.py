import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

# Define the GraphSAGE model class
class GraphSAGEModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, normalize=True):
        super(GraphSAGEModel, self).__init__()

        self.num_layers = num_layers

        # List to store the GraphSAGE convolutional layers
        self.conv_layers = nn.ModuleList()

        # Input layer
        self.conv_layers.append(SAGEConv(num_features, hidden_dim))

        # Intermediate layers
        for _ in range(num_layers - 2):
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