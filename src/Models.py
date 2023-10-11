import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphUNet
from torch_geometric.utils import dropout_edge
      
# Graph U-Net class
class GUNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, final_act):
        super().__init__()

        self.act_middle = torch.nn.functional.relu
        # activation = torch.nn.Linear(1, 1); activation(x)
        # torch.sigmoid
        # torch.nn.Sigmoid
        # F.log_softmax(x, dim=1)     # original version
        # torch.nn.Linear(1, 1)
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

        return self.act_final(x)

    def _log_network(self):
        middle = self.act_middle.__name__
        final = self.act_final.__module__
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