import torch
import torch_geometric

from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GINConv, GCNConv


class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels) -> None:
        super(GCN, self).__init__()
        
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
    def forward(self, x, edge_index):
        
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        
        return x
        



# class BaseModel(torch.nn.Module):
#     def __init__(self, dataset, hidden_channels):
#         super(BaseModel, self).__init__()
#         input_dim = dataset.num_node_features
#         # weight seed
#         torch.manual_seed(42)
        
#         self.conv1 = GINConv(Sequential(Linear(input_dim, hidden_channels),
#                                         ReLU(), Linear(hidden_channels, hidden_channels)))
#         self.conv2 = GINConv(Sequential(Linear(hidden_channels, hidden_channels),
#                                         ReLU(), Linear(hidden_channels, hidden_channels)))
        
        
#         self.lin = Linear(hidden_channels, hidden_channels)
#         self.lin2 = Linear(hidden_channels, dataset.num_classes)
    
#     def forward(self, x, edge_index, batch):
        
#         x = self.conv1()