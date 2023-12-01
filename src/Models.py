import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphUNet
from torch_geometric.utils import dropout_edge

import customGraphUNet

# Graph U-Net class
class GUNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, final_act):
        super().__init__()

        self.act_middle = torch.nn.functional.relu
        self.act_final = final_act
        
        # Max pooling
        #K = 1
        #pool_ratios = [K / num_nodes]
        
        # Top-K pooling
        K = 2000
        #pool_ratios = [K / num_nodes, 0.5]
        pool_ratios = [0.25, 0.5]
        
        self.unet = GraphUNet(in_channels, hidden_channels, out_channels, depth=3, pool_ratios=pool_ratios, act=self.act_middle)
                #customGraphUNet.GraphUNet
        
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
