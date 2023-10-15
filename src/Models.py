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
        
# Graph UNet with batch normilization after each convulational layer
# We use subclassing to add the batchnorma
class GUNeteBatch(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super(GUNetBatch, self).__init__()
        self.depth = depth
        
        # Create a list to hold the encoder layers
        self.encoders = nn.ModuleList()
        
        # Create a list to hold the decoder layers
        self.decoders = nn.ModuleList()
        
        # Create BatchNorm layers for each layer
        self.batch_norms = nn.ModuleList()
        
        # Encoder (down-sampling)
        for i in range(self.depth):
            self.encoders.append(nn.Sequential(
                GraphUNet(in_channels, out_channels, depth=1),
                nn.BatchNorm1d(out_channels),  # BatchNorm after each encoder
                nn.ReLU()
            ))
            in_channels = out_channels
        
        # Decoder (up-sampling)
        for i in range(self.depth - 1):
            self.decoders.append(nn.Sequential(
                GraphUNet(in_channels * 2, out_channels, depth=1),
                nn.BatchNorm1d(out_channels),  # BatchNorm after each decoder
                nn.ReLU()
            ))
            in_channels = out_channels
        
        # Final decoder without the BatchNorm
        self.decoders.append(nn.Sequential(
            GraphUNet(in_channels * 2, out_channels, depth=1),
            nn.ReLU()
        ))
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        xs = []
        
        # Encoder (down-sampling)
        for i in range(self.depth):
            x = self.encoders[i](x, edge_index)
            xs.append(x)
        
        # Decoder (up-sampling)
        for i in range(self.depth - 1, -1, -1):
            x = self.decoders[i](x, torch.cat([x, xs[i]], dim=1), edge_index)
        
        return x
    
# Graph Unet with batch normilization
class GraphUNetWithBN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth):
        super(GraphUNetWithBN, self).__init__()
        self.depth = depth

        # Create a list to hold the encoder layers
        self.encoders = nn.ModuleList()

        # Create a list to hold the decoder layers
        self.decoders = nn.ModuleList()

        # Create BatchNorm layers for each layer
        self.batch_norms = nn.ModuleList()

        # Encoder (down-sampling)
        for i in range(self.depth):
            self.encoders.append(GraphUNet(in_channels, hidden_channels, out_channels, depth=1))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))  # BatchNorm after each encoder
            in_channels = hidden_channels

        # Decoder (up-sampling)
        for i in range(self.depth - 1):
            self.decoders.append(GraphUNet(in_channels * 2, hidden_channels, out_channels, depth=1))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))  # BatchNorm after each decoder
            in_channels = hidden_channels

        # Final decoder without the BatchNorm
        self.decoders.append(GraphUNet(in_channels * 2, out_channels, out_channels, depth=1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        xs = []

        # Encoder (down-sampling)
        for i in range(self.depth):
            x = self.encoders[i](x, edge_index)
            x = self.batch_norms[i](x)
            xs.append(x)

        # Decoder (up-sampling)
        for i in range(self.depth - 1, -1, -1):
            x = self.decoders[i](x, torch.cat([x, xs[i]], dim=1), edge_index)

        return x