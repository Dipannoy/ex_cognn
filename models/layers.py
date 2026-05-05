from typing import Callable, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import remove_self_loops, add_remaining_self_loops

import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair

def get_virtualnode_mlp(emb_dim: int) -> Module:
    return torch.nn.Sequential(Linear(emb_dim, 2 * emb_dim), BatchNorm1d(2 * emb_dim), ReLU(),
                               Linear(2 * emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU())

# class WeightedGINConv(Module):
#     def __init__(
#         self,
#         apply_func=None,
#         aggregator_type="sum",
#         init_eps=0,
#         learn_eps=False,
#         activation=None,
#     ):
#         super(WeightedGINConv, self).__init__()
#         self.apply_func = apply_func
#         self._aggregator_type = aggregator_type
#         self.activation = activation
#         if aggregator_type not in ("sum", "max", "mean"):
#             raise KeyError(
#                 "Aggregator type {} not recognized.".format(aggregator_type)
#             )
#         # to specify whether eps is trainable or not.
#         if learn_eps:
#             self.eps = Parameter(torch.FloatTensor([init_eps]))
#         else:
#             self.register_buffer("eps", torch.FloatTensor([init_eps]))

#     def forward(self, graph, feat, edge_weight=None, attr_act="add"):
#         _reducer = getattr(fn, self._aggregator_type)

#         with graph.local_scope():
#             feat_src, feat_dst = expand_as_pair(feat, graph)
#             graph.srcdata["h"] = feat_src

#             aggregate_fn = fn.copy_u("h", "m")
#             if graph.edata["edge_attr"] is not None:
#                 if attr_act == "add":
#                     assert feat.shape[-1] == graph.edata["edge_attr"].shape[-1], \
#                     f"last feat dim {feat.shape[-1]} != last edge_attr dim {graph.edata['edge_attr'].shape[-1]"
#                     graph.edata["edge_attr"] = graph.srcdata["h"] + graph.edata["edge_attr"] 
#                 elif attr_act == "cat":
#                     graph.edata["edge_attr"] = torch.cat([graph.edata["edge_attr"], graph.srcdata["h"]], dim=-1)
#                 elif attr_act == "mul":
#                     pass
                
#             if edge_weight is not None:
#                 assert edge_weight.shape[0] == graph.num_edges()
#                 if graph.edata["edge_attr"] is not None:
#                     graph.edata["_edge_weight"] = edge_weight * graph.edata["edge_attr"]
#                 else:
#                     graph.edata["_edge_weight"] = edge_weight
#                 aggregate_fn = fn.u_mul_e("h", "_edge_wieght", "m")

#             graph.update_all(aggregate_fn, _reducer("m", "neigh"))
#             rst = (1 + self.eps) * feat_dst + graph.dstdata["neigh"]
#             if self.apply_func is not None:
#                 rst = self.apply_func(rst)
#             # activation
#             if self.activation is not None:
#                 rst = self.activation(rst)
#             return rst


class WeightedGINEConv(Module):
    """GINE: Graph Isomorphism Network (GIN) with Edge Features"""
    def __init__(self, apply_func=None, init_eps=0, learn_eps=False):
        super(WeightedGINEConv, self).__init__()
        self.apply_func = apply_func
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))

    def message(self, edges):
        r"""User-defined Message Function"""
        if "we" in edges.data.keys():
            we = edges.data["we"]
        else:
            we = 1.
        return {"m": we * F.relu(edges.src["hn"] + edges.data["he"])}

    def forward(self, graph, node_feat, edge_feat, edge_weight=None):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(node_feat, graph)
            graph.srcdata["hn"] = feat_src
            graph.edata["he"] = edge_feat
            if edge_weight is not None:
                graph.edata["we"] = edge_weight
            graph.update_all(self.message, fn.sum("m", "neigh"))
            rst = (1 + self.eps) * feat_dst + graph.dstdata["neigh"]
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            return rst
            