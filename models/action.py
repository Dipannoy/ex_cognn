import torch.nn as nn
from torch import Tensor
#from helpers.classes import ActionNetArgs
import dgl

from .layers import WeightedGINEConv

# class ActionNet(nn.Module):
#     def __init__(self, action_args: ActionNetArgs):
#         """
#         Create a model which represents the agent's policy.
#         """
#         super().__init__()
#         self.num_layers = action_args.num_layers
#         self.net = action_args.load_net()
#         self.dropout = nn.Dropout(action_args.dropout)
#         self.act = action_args.act_type.get()

#     def forward(self, g: dgl.DGLGraph, x: Tensor, y: Tensor) -> Tensor:
#         edge_attrs = [g.edata['env_edge_attr']] + (self.num_layers - 1) * [g.edata['act_edge_attr']]
#         with g.local_scope():
#             for idx, (edge_attr, layer) in enumerate(zip(edge_attrs[:-1], self.net[:-1])):
#                 #g.edata['edge_attr'] = edge_attr
#                 x = layer(g, x, edge_feat=edge_attr)
#                 x = self.dropout(x)
#                 x = self.act(x)
#             #g.edata['edge_attr'] = edge_attrs[-1]
#             x = self.net[-1](g, x, edge_feat=edge_attr)
#         return x


class ActionNetForTest(nn.Module):
    def __init__(self, action_args):
        """
        Create a model which represents the agent's policy.
        """
        super().__init__()
        self.num_layers = action_args.num_layers
        self.nets = [] 
        for _ in range(self.num_layers):
            self.nets.append(WeightedGINEConv())
        self.act = nn.ReLU(inplace=True)

    def forward(self, g: dgl.DGLGraph, x: Tensor, y: Tensor) -> Tensor:
        for layer in self.nets[:-1]:
            x = layer(g, x, y)
            x = self.act(x)
        x = self.nets[-1](g, x, y)
        return x
