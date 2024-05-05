from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear

import torch
n_features = 9

class GCNModel(torch.nn.Module):
    def __init__(self):
        super(GCNModel, self).__init__()
        torch.manual_seed(12)
        self.input = None     
        self.edge_calc = None     
        self.conv1 = GCNConv(n_features, 8)
        self.conv2 = GCNConv(8, 16)
        self.conv3 = GCNConv(16, 32)
        self.pooling = global_mean_pool
        self.linear = Linear(32, 2) 


    def forward(self, x, edge_index, batch):

        self.input = x   

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.pooling(x, batch=batch)
                
        x = self.linear(x)
        return x