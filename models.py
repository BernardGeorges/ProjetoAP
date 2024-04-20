from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear

import torch
n_features = 15

class GCNModel(torch.nn.Module):
    def __init__(self):
        super(GCNModel, self).__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(n_features, 8)
        self.conv2 = GCNConv(8, 16)
        self.conv3 = GCNConv(16, 32)
        self.pooling = global_mean_pool
        self.linear = Linear(32, 1)

    # PERGUNTAR COMO CALCULAR OS GRADIENTES DOS NODOS E DAS LIGACOES <----- 

    def forward(self, x, edge_index, edge_attr, batch):

        h1 = F.relu(self.conv1(x, edge_index, edge_attr))
        h2 =  F.relu(self.conv2(x, edge_index, edge_attr))

        h3 = self.conv3(x, edge_index, edge_attr)        
        h4 = self.pooling(h3,batch=batch)
        x = self.linear(h4)

        return x
    
