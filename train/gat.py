import torch
from torch.nn import Linear, ELU, Sequential, ReLU, Sigmoid, Softmax
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool, GATv2Conv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GATReg(torch.nn.Module):
    def __init__(self,
                 num_features,
                 out_channels,
                 heads,
                 bias=True,
                 dropout=0.06):
        
        # Init parent
        super(GATReg, self).__init__()
        torch.manual_seed(42)

        self.gat1 = GATv2Conv(
            in_channels=num_features[0],
            out_channels=out_channels[0],
            heads=heads[0],
            concat=True,
            dropout_prob=dropout,
            bias=bias,
            return_attention_weights=False)
        
        self.activation1 = ReLU()

        self.gat2 = GATv2Conv(
            in_channels=num_features[1],
            out_channels=out_channels[1],
            heads=heads[1],
            concat=True,
            dropout_prob=dropout,
            bias=bias,
            return_attention_weights=False)
        
        self.activation2 = ReLU()

        self.finalGat = GATv2Conv(
            in_channels=out_channels[-1] * heads[-1],
            out_channels=1,
            heads=1,
            concat=False,
            dropout_prob=dropout,
            bias=bias,
            return_attention_weights=False
        )

        self.out = Sigmoid()

    def forward(self, x, edge_index):

        out = self.gat1(x, edge_index)
        out = self.activation1(out)
        #out = F.dropout(out, p=0.2, training=self.training)

        out = self.gat2(out, edge_index)
        out = self.activation2(out)
        #out = F.dropout(out, p=0.2, training=self.training)

        out = self.finalGat(out, edge_index)
        #out = self.finalActivation(out)
        #out = F.dropout(out, p=0.2, training=self.training)

        out = out.view(x.shape[0])
        out = self.out(out)

        return out
    
class GATRegGraphPrediction(torch.nn.Module):
    def __init__(self,
                 num_features,
                 out_channels,
                 heads,
                 bias=True,
                 dropout=0.06,
                 outputVal=1):
        
        self.outputVal = outputVal
        
        # Init parent
        super(GATRegGraphPrediction, self).__init__()
        torch.manual_seed(42)

        self.gat1 = GATv2Conv(
            in_channels=num_features[0],
            out_channels=out_channels[0],
            heads=heads[0],
            concat=True,
            dropout_prob=dropout,
            bias=bias,
            return_attention_weights=False)

        self.activation1 = ELU()

        self.gat2 = GATv2Conv(
            in_channels=num_features[1],
            out_channels=out_channels[1],
            heads=heads[1],
            concat=True,
            dropout_prob=dropout,
            bias=bias,
            return_attention_weights=False)

        self.activation2 = ELU()
        
        self.finalGat = GATv2Conv(
            in_channels=out_channels[-1] * heads[-1],
            out_channels=outputVal,
            heads=1,
            concat=False,
            dropout_prob=dropout,
            bias=bias,
            return_attention_weights=False
        )
 
        self.out = Sigmoid()

    def forward(self, x, edge_index):

        out = self.gat1(x, edge_index)
        out = self.activation1(out)

        out = self.gat2(out, edge_index)
        out = self.activation2(out)

        out = self.finalGat(out, edge_index)

        out = self.out(out)
        out = out.view((x.shape[0], self.outputVal))
        
        return out
    
class GATBin(torch.nn.Module):
    def __init__(self,
                 num_features,
                 out_channels,
                 heads,
                 bias=True,
                 dropout=0.06,
                 outputVal=1):
         
        # Init parent
        super(GATBin, self).__init__()
        torch.manual_seed(42)

        self.gat1 = GATv2Conv(
            in_channels=num_features[0],
            out_channels=out_channels[0],
            heads=heads[0],
            concat=True,
            dropout_prob=dropout,
            bias=bias,
            return_attention_weights=False)

        self.activation1 = ReLU()
        
        self.gat2 = GATv2Conv(
            in_channels=num_features[1],
            out_channels=2,
            heads=heads[1],
            concat=True,
            dropout_prob=dropout,
            bias=bias,
            return_attention_weights=False)
        
        self.activation2 = ReLU()
        
        """self.gat3 = GATv2Conv(
            in_channels=num_features[2],
            out_channels=out_channels[2],
            heads=heads[2],
            concat=True,
            dropout_prob=dropout,
            bias=bias,
            return_attention_weights=False)"""

        self.finalGat = GATv2Conv(
            in_channels=out_channels[-1] * heads[-1],
            out_channels=outputVal,
            heads=1,
            concat=False,
            dropout_prob=dropout,
            bias=bias,
            return_attention_weights=False
        )
        
        self.activation3 = ReLU()
 
    def forward(self, x, edge_index):

        out = self.gat1(x, edge_index)
        out = self.activation1(out)

        out = self.gat2(out, edge_index)
        out = self.activation2(out)

        """out = self.gat3(out, edge_index)
        out = self.activation(out)"""

        out = self.finalGat(out, edge_index)
        out = self.activation3(out)
  
        out = F.softmax(out, dim=1)
        return out