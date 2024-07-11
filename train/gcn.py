import torch
from torch.nn import Linear, Sigmoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GCN(torch.nn.Module):
    def __init__(self, num_features, embedding_size):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, 1)

        # Output layer
        self.out = Sigmoid()

    def forward(self, x, edge_index):

        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)

        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)

        hidden = hidden.view(x.shape[0])

        out = self.out(hidden)

        return out