import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphUNet
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import dropout_edge
import Loss

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
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, num_nodes, final_act):
        super(GCNModel, self).__init__()
        
        self.num_layers = num_layers
        self.act_middle = torch.nn.functional.relu
        self.act_final = final_act
        
        pool_ratios = [2000 / num_nodes, 0.5]

        # List to store the GraphSAGE convolutional layers
        self.conv_layers = nn.ModuleList()

        # Input layer
        self.conv_layers.append(GCNConv(num_features, hidden_dim, pool_ratios=pool_ratios, act=self.act_middle))
        #nn.init.kaiming_uniform_(self.conv_layers.weight, mode='fan_in', nonlinearity='relu')

        # Apply He initialization to the underlying linear transformations
        # for i in range(len(self.conv_layers)):
        #     nn.init.kaiming_uniform_(self.conv_layers[i].lin_l.weight)
        #     nn.init.kaiming_uniform_(self.conv_layers[i].lin_r.weight)

        # Intermediate layers
        for _ in range(1, num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim, pool_ratios=pool_ratios, act=self.act_middle))

        # Output layer
        self.conv_layers.append(GCNConv(hidden_dim, num_classes, pool_ratios=pool_ratios, act=self.act_final))

        #self.normalize = normalize

        
        
    def forward(self, data):
        edge_index, _ = dropout_edge(data.edge_index, p=0.2,
                                     force_undirected=True,
                                     training=self.training)
        x = F.dropout(data.x, p=0.92, training=self.training)
        for layer in self.conv_layers:
            x = layer(x, edge_index)

            act = None
            if self.act_final.__name__ == "softmax":
                act = self.act_final(x, dim=1)
            elif self.act_final.__name__ == "sigmoid":
                act = self.act_final(x)
        return act

    def _log_network(self):
        middle = self.act_middle.__name__
        final = self.act_final.__name__
        print(f"GCN instantiated!\n\tMiddle act: {middle}\n\tFinal act: {final}")   
        
# Define the GraphSAGE model class
class GraphSAGEModel(nn.Module): 
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, num_nodes, final_act):#, normalize=True, dropout=0.0):
        super(GraphSAGEModel, self).__init__()

        self.num_layers = num_layers
        self.act_middle = torch.nn.functional.relu
        self.act_final = final_act
        
        pool_ratios = [2000 / num_nodes, 0.5]

        # List to store the GraphSAGE convolutional layers
        self.conv_layers = nn.ModuleList()

        # Input layer
        self.conv_layers.append(SAGEConv(num_features, hidden_dim, pool_ratios=pool_ratios, act=self.act_middle))
        #nn.init.kaiming_uniform_(self.conv_layers.weight, mode='fan_in', nonlinearity='relu')

        # Apply He initialization to the underlying linear transformations
        # for i in range(len(self.conv_layers)):
        #     nn.init.kaiming_uniform_(self.conv_layers[i].lin_l.weight)
        #     nn.init.kaiming_uniform_(self.conv_layers[i].lin_r.weight)

        # Intermediate layers
        for _ in range(1, num_layers - 1):
            self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim, pool_ratios=pool_ratios, act=self.act_middle))

        # Output layer
        self.conv_layers.append(SAGEConv(hidden_dim, num_classes, pool_ratios=pool_ratios, act=self.act_final))

        #self.normalize = normalize

        
        
    def forward(self, data):
        edge_index, _ = dropout_edge(data.edge_index, p=0.2,
                                     force_undirected=True,
                                     training=self.training)
        x = F.dropout(data.x, p=0.92, training=self.training)
        for layer in self.conv_layers:
            x = layer(x, edge_index)

            act = None
            if self.act_final.__name__ == "softmax":
                act = self.act_final(x, dim=1)
            elif self.act_final.__name__ == "sigmoid":
                act = self.act_final(x)
        return act

    def _log_network(self):
        middle = self.act_middle.__name__
        final = self.act_final.__name__
        print(f"Graph Sage instantiated!\n\tMiddle act: {middle}\n\tFinal act: {final}")    
        
    
    ## Possible improvements points
    # Number of layer might significantly impact its performance
    # Different hidden dims (e.g., 64, 128, 256)
    # Include or exclude L2 normalization - try
    # Number of Aggregation Neighbors (num_samples) - accuracy vs performance
    #Batch size, epoch, optimizer loss

    
# Graph attention model design    
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, edge_index):
        edge_index, _ = dropout_edge(edge_index, p=self.dropout, force_undirected=True, training=self.training)
        x = F.dropout(x, p=0.92, training=self.training)
        h = torch.matmul(x, self.W)
        N = h.size()[0]

        row, col = edge_index
        a_input = torch.cat([h[row], h[col]], dim=1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(1))

        # Normalize attention scores
        edge_weights = F.softmax(e, dim=0)

        # Aggregate neighborhood features using the normalized attention scores
        # h_prime = scatter_add(edge_weights * h[col], row, dim=0, dim_size=N)
        h_prime = scatter_add(edge_weights.view(-1, 1) * h[col], row, dim=0, dim_size=N)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden, n_layers, dropout, alpha):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GraphAttentionLayer(n_features, n_hidden, dropout, alpha))
        #nn.init.kaiming_uniform_(self.conv_layers.weight, mode='fan_in', nonlinearity='relu')

        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(GraphAttentionLayer(n_hidden, n_hidden, dropout, alpha))

        # Output layer
        self.layers.append(GraphAttentionLayer(n_hidden, n_classes, dropout, alpha, concat=False))

    def forward(self, data):
        edge_index, _ = dropout_edge(data.edge_index, p=0.2,
                                     force_undirected=True,
                                     training=self.training)
        x = F.dropout(data.x, p=0.92, training=self.training)
        for layer in self.layers:
            x = layer(x, edge_index)
        return F.log_softmax(x, dim=1)