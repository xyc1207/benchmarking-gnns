import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_layer import GATLayer, CustomGATLayer, CustomGATLayerEdgeReprFeat, CustomGATLayerIsotropic
from layers.gat_layer import GATLayerV1, GATLayerV2, GATLayerV3, GATLayerV4, GATLayerV5, GATLayerV6, GATLayerV7
from layers.mlp_readout_layer import MLPReadout

class GATNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']

        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.device = net_params['device']

        self.layer_type = {
            "dgl": GATLayer,
            "dgl-v1": GATLayerV1,
            "dgl-v2": GATLayerV2,
            "dgl-v3": GATLayerV3,
            "dgl-v4": GATLayerV4,
            "dgl-v5": GATLayerV5,
            "dgl-v6": GATLayerV6,
            "dgl-v7": GATLayerV7,
            "edgereprfeat": CustomGATLayerEdgeReprFeat,
            "edgefeat": CustomGATLayer,
            "isotropic": CustomGATLayerIsotropic,
        }.get(net_params['layer_type'], GATLayer)

        self.embedding_h = torch.nn.Embedding(in_dim, hidden_dim * num_heads).to(self.device)
        torch.nn.init.xavier_uniform_(self.embedding_h.weight)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([self.layer_type(hidden_dim * num_heads, hidden_dim, num_heads,
                                                     dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(self.layer_type(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(2*out_dim, 1)
        
    def forward(self, g):
        h = self.embedding_h.weight
        h = self.in_feat_dropout(h)

        for conv in self.layers:
            h = conv(g, h)

        g.ndata['h'] = h
        
        return h
    
    def edge_predictor(self, h_i, h_j):
        x = torch.cat([h_i, h_j], dim=1)
        x = self.MLP_layer(x)
        
        return torch.sigmoid(x)
    
    def loss(self, pos_out, neg_out):
        pos_loss = -torch.log(pos_out + 1e-15).mean()  # positive samples
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()  # negative samples
        loss = pos_loss + neg_loss
        
        return loss
