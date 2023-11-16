import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphUNet
from torch_geometric.nn import SAGEConv
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
    
# Definition of an Ensemble class to combine predictions of the three models
class EnsembleModel(nn.Module):
    def __init__(self, GUNet, GCNModel, GraphSAGEModel, weights=None):
        super(EnsembleModel, self).__init__()
        self.GUNet = GUNet
        self.GCNModel = GCNModel
        self.GraphSAGEModel = GraphSAGEModel

        # Define weights for each model in the ensemble
        if weights is None:
            self.weights = nn.Parameter(torch.ones(3))  # Default: equal weights
        else:
            self.weights = nn.Parameter(torch.Tensor(weights))
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Apply the Graph U-Net to the feature vectors
        x_unet = self.GUNet(data)
        # Apply the GCN to the graph and feature vectors
        x_gcn = self.GCNModel(data)
        # Apply the GraphSAGE to the graph and feature vectors
        x_sage = self.GraphSAGEModel(data)
        x = (self.weights[0] * x_unet + self.weights[1] * x_gcn + self.weights[2] * x_sage) / sum(self.weights)
        return x
        
#Implementation of voting ensemble
class VotingEnsemble(nn.Module):
    def __init__(self, models):
        super(VotingEnsemble, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, data):
        predictions = [model(data) for model in self.models]
        # Soft voting: Combine predictions using element-wise averaging
        ensemble_output = torch.stack(predictions, dim=0).mean(dim=0)
        return ensemble_output
    
# Focal loss fusion ensemble. 
#Apply focal loss to each model's predictions before fusion to down-weights the background class  and up-weight cyclonic and anticyclonic eddies classes to counter the imbalanced datasets.
class EnsembleFocal(nn.Module):
    def __init__(self, GUNet, GCNModel, GraphSAGEModel, weights=None, focal_loss_params=None):
        super(EnsembleFocal, self).__init__()
        self.GUNet = GUNet
        self.GCNModel = GCNModel
        self.GraphSAGEModel = GraphSAGEModel

        if weights is None:
            self.weights = nn.Parameter(torch.ones(3))
        else:
            self.weights = nn.Parameter(torch.Tensor(weights))

        # Focal loss parameters
        self.focal_loss_params = focal_loss_params
        self.focal_loss = Loss.FocalLoss(**focal_loss_params) if focal_loss_params else None

    def forward(self, data):
        x_unet = self.GUNet(data)
        x_gcn = self.GCNModel(data)
        x_sage = self.GraphSAGEModel(data)

        # Combine model outputs with weights
        combined_output = (self.weights[0] * x_unet + self.weights[1] * x_gcn + self.weights[2] * x_sage) / sum(self.weights)

        # Apply focal loss if specified
        if self.focal_loss:
            target = data.y  # Assuming the ground truth labels are stored in data.y
            loss = self.focal_loss(combined_output, target)
            return combined_output, loss
        else:
            return combined_output
