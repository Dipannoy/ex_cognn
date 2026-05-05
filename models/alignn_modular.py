"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from typing import Tuple, Union

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling

# from dgl.nn.functional import edge_softmax
#from pydantic.typing import Literal
from torch import nn
from torch.nn import functional as F

#from alignn.models.utils import RBFExpansion
#from alignn.utils import BaseSettings

from .utils import batch_node_subgraph, batch_edge_subgraph, RBFExpansion
#from .layers import WeightedGINEConv
#from .action import ActionNetForTest


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_weights: torch.Tensor = None,
        node_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        # create a subgraph
        if node_mask is not None:
            g = batch_node_subgraph(g, node_mask)
            edge_mask = g.edata[dgl.EID]
        else:
            node_mask = torch.ones((node_feats.shape[0],)).bool().to(m.device)
            edge_mask = torch.ones((edge_feats.shape[0],)).bool().to(m.device)

        # mask edges depending on node state
        edge_weights = edge_weights.view(-1,1) if edge_weights is not None else 1.
        g.edata["sigma"] = edge_weights * torch.sigmoid(m[edge_mask])
        g.ndata["Bh"] = self.dst_update(node_feats[node_mask])
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # 1st version
        # h = self.src_update(node_feats[node_mask]) + g.ndata.pop("h")
        # x = node_feats.clone()
        # x[node_mask] = h

        # 2nd version
        x = self.src_update(node_feats)
        h = torch.zeros_like(x)
        h[node_mask] = g.ndata.pop("h")
        x = x + h

        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        # This current one updates node and edge of the whole graph.
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))       
        # Should we update only node and edge in the subgraph?
        # x[node_mask] = F.silu(self.bn_nodes(x[node_mask]))
        # y[edge_mask] = F.silu(self.bn_edges(m[edge_mask])) 

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask_before_next: bool=False,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)
        self.mask_before_next = mask_before_next

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        w_bonds: torch.Tensor,
        w_triplets: torch.Tensor,
        keep_bond_forcing: bool=False,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        w_bonds: edge weights
        w_triplets: edge pair weights
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y, w_bonds)
        if self.mask_before_next:
            m = m * w_bonds

        # Edge-gated graph convolution update on crystal graph
        if keep_bond_forcing:
            ones = torch.ones(g.num_edges(), requires_grad=False).bool().to(x.device)
            y, z = self.edge_update(lg, m, z, w_triplets, ones)
        else:
            y, z = self.edge_update(lg, m, z, w_triplets, w_bonds.bool())
        if self.mask_before_next:
            z = z * w_triplets

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


# class CoALIGNN(nn.Module):
#     """Atomistic Line graph network.

#     Chain alternating gated graph convolution updates on crystal graph
#     and atomistic line graph.
#     """

#     def __init__(self, config):
#         """Initialize class with number of input features, conv layers."""
#         super().__init__()
#         # print(config)
#         self.config = config
#         self.classification = config.classification

#         input_node_dim = 0
#         if config.atom_input_features > 0:
#             self.atom_embedding = MLPLayer(
#                 config.atom_input_features, config.hidden_features
#             )
#             input_node_dim += config.hidden_features

#         if config.extra_atom_features is not None:
#             if isinstance(config.extra_atom_features, int):
#                 self.config.extra_atom_features = {"extra_atom_features": config.extra_atom_features}
            
#             self.extra_atom_embedding = nn.ModuleDict()
#             for k, v in self.config.extra_atom_features.items():
#                 self.extra_atom_embedding[k] = MLPLayer(
#                     v, config.hidden_features
#                 )
#                 input_node_dim += config.hidden_features

#             self.squeeze_atom_embedding = MLPLayer(
#                     input_node_dim, config.hidden_features
#                 )
#             input_node_dim = config.hidden_features

#         self.edge_embedding = nn.Sequential(
#             RBFExpansion(
#                 vmin=0,
#                 vmax=8.0,
#                 bins=config.edge_input_features,
#             ),
#             MLPLayer(config.edge_input_features, config.embedding_features),
#             MLPLayer(config.embedding_features, input_node_dim),
#         )
#         self.angle_embedding = nn.Sequential(
#             RBFExpansion(
#                 vmin=-1,
#                 vmax=1.0,
#                 bins=config.triplet_input_features,
#             ),
#             MLPLayer(config.triplet_input_features, config.embedding_features),
#             MLPLayer(config.embedding_features, input_node_dim),
#         )

#         self.alignn_layers = nn.ModuleList(
#             [
#                 ALIGNNConv(
#                     input_node_dim,
#                     input_node_dim,
#                     config.mask_before_next,
#                 )
#                 for idx in range(config.alignn_layers)
#             ]
#         )
#         self.gcn_layers = nn.ModuleList(
#             [
#                 EdgeGatedGraphConv(
#                     input_node_dim, input_node_dim
#                 )
#                 for idx in range(config.gcn_layers)
#             ]
#         )

#         self.is_hard = config.is_hard
#         self.mask_triplets = config.mask_triplets
#         self.num_act_graph_layers = config.action_args.num_act_graph_layers
#         if self.num_act_graph_layers > 0:
#             # self.act_atom_embedding = MLPLayer(
#             #     input_node_dim, input_node_dim
#             # )     
#             # self.act_edge_embedding = nn.Sequential(
#             #     RBFExpansion(
#             #         vmin=0,
#             #         vmax=8.0,
#             #         bins=config.edge_input_features,
#             #     ),
#             #     MLPLayer(config.edge_input_features, config.embedding_features),
#             #     MLPLayer(config.embedding_features, config.hidden_features),
#             # )
#             self.act_edge_embedding = MLPLayer(
#                 config.hidden_features, config.hidden_features
#             )
#             self.in_act_graph_net = ActionNet(self.num_act_graph_layers, 
#                                               input_node_dim, 2)
#             self.out_act_graph_net = ActionNet(self.num_act_graph_layers, 
#                                                input_node_dim, 2)
            
#         self.num_act_linegraph_layers = config.action_args.num_act_linegraph_layers
#         if self.num_act_linegraph_layers > 0:
#             self.in_act_linegraph_net = ActionNet(self.num_act_linegraph_layers, 
#                                               input_features, 2)
#             self.out_act_linegraph_net = ActionNet(self.num_act_graph_layers, 
#                                                input_features, 2)

#         self.learn_temp = config.learn_temp
#         self.temp = config.temp

#         self.readout = AvgPooling()
#         self.readout_feat = AvgPooling()
#         if self.classification:
#             self.fc = nn.Linear(config.hidden_features, config.num_classes)
#             self.softmax = nn.LogSoftmax(dim=1)
#         else:
#             self.fc = nn.Linear(config.hidden_features, config.output_features)

#         # if config.extra_features > 0:
#         #     # Credit for extra_features work:
#         #     # Gong et al., https://doi.org/10.48550/arXiv.2208.05039
#         #     self.extra_feature_embedding = MLPLayer(
#         #         config.extra_features, config.extra_features
#         #     )
#         hidden_dim = input_node_dim
#         if config.extra_features is not None:
#             if isinstance(config.extra_features, int):
#                 self.config.extra_features = {"extra_features": config.extra_features}            
#             self.extra_feature_embedding = nn.ModuleDict()
#             for k, v in self.config.extra_features.items():
#                 self.extra_feature_embedding[k] = MLPLayer(v, v)  
#                 hidden_dim += v

#         if config.global_features is not None:
#             if isinstance(config.global_features, int):
#                 self.config.global_features = {"global_features": config.global_features}
            
#             self.global_feature_embedding = nn.ModuleDict()
#             for k, v in self.config.global_features.items():
#                 if config.embed_global_features:
#                     self.global_feature_embedding[k] = nn.Sequential(
#                         MLPLayer(v, config.hidden_features), 
#                         # nn.Linear(config.hidden_features, config.hidden_features),
#                         )                    
#                     hidden_dim += config.hidden_features
#                 else:
#                     self.global_feature_embedding[k] = nn.Identity()
#                     hidden_dim += v
            
#         self.fc = nn.ModuleList()
#         if (config.extra_features is not None) or (config.global_features is not None):
#             for _ in range(2):
#                 self.fc.append(MLPLayer(hidden_dim, hidden_dim))
#         self.fc.append(nn.Linear(hidden_dim, config.output_features))

#         self.link = None
#         self.link_name = config.link
#         if config.link == "identity":
#             self.link = lambda x: x
#         elif config.link == "log":
#             self.link = torch.exp
#             avg_gap = 0.7  # magic number -- average bandgap in dft_3d
#             self.fc.bias.data = torch.tensor(
#                 np.log(avg_gap), dtype=torch.float
#             )
#         elif config.link == "logit":
#             self.link = torch.sigmoid

#     def forward(
#         self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph],
#         keep_bond_forcing: bool=False, context=torch.enable_grad(), is_hard=False,
#     ):
#         """ALIGNN : start with `atom_features`.

#         x: atom features (g.ndata)
#         y: bond features (g.edata and lg.ndata)
#         z: angle features (lg.edata)
#         """
#         if len(self.alignn_layers) > 0:
#             g, lg = g
#             lg = lg.local_var()

#             # angle features (fixed)
#             z = self.angle_embedding(lg.edata.pop("h"))
#         else:
#             g, lg = g

#         # In an updated alignn code, extra node-level features are 
#         # embeded using mlp and then pooled to get graph-level features. 
#         # before concatenating with the graph-level features pooled
#         # from gnn.
#         # if self.config.extra_features != 0:
#         #     features = g.ndata["extra_features"]
#         #     features = self.extra_feature_embedding(features)
#         if self.config.extra_features is not None:
#             features = []
#             for k, layer in self.extra_feature_embedding.items():
#                 h_extra = g.ndata.pop(k)
#                 h_extra = layer(h_extra)
#                 features.append(h_extra)
#             features = torch.cat(features, dim=-1)

#         g = g.local_var()
        
#         # initial node features: atom feature network...
#         if self.config.atom_input_features > 0:
#             x = g.ndata.pop("atom_features")
#             x = self.atom_embedding(x)
#         if self.config.extra_atom_features is not None:
#             atom_feats = [x] if self.config.atom_input_features > 0 else []
#             for k, layer in self.extra_atom_embedding.items():
#                 f = g.ndata.pop(k)
#                 atom_feats.append(layer(f))
#             x = torch.cat(atom_feats, dim=-1)
#             x = self.squeeze_atom_embedding(x)

#         # initial bond features
#         bondlength = torch.norm(g.edata.pop("r"), dim=1)
#         y = self.edge_embedding(bondlength)

#         survived_ratios = {'g': [], 'lg': [], 'slg': [], 'gcn': []}
        
#         for alignn_layer in self.alignn_layers:
#             # Policy net for node states on Graph.
#             if self.num_act_graph_layers > 0:
#                 x_act = x
#                 # x_act = self.act_atom_embedding(x)
#                 y_act = self.act_edge_embedding(y)

#                 with context:
#                     in_bond_logits, in_trip_logits = self.in_act_graph_net(g, x_act, y_act)  # (N, 2)
#                     out_bond_logits, out_trip_logits = self.out_act_graph_net(g, x_act, y_act)  # (N, 2)
    
#                     temp = self.temp_model(g=g, edge_attr=y) if self.learn_temp else self.temp
#                     w_bonds = self.get_edge_logits(g, in_bond_logits, out_bond_logits, temp, is_hard)
#                     survived_ratios['g'].append((w_bonds.detach().sum() / g.num_edges()).item())
    
#                     # not sure which one is better.
#                     if self.mask_triplets:
#                         # Policy net for node states on Line Graph.
#                         if keep_bond_forcing:
#                             slg = batch_node_subgraph(lg, torch.ones(g.num_edges()).to(x.device))
#                         else:
#                             slg = batch_node_subgraph(lg, w_bonds.bool())
#                         w_triplets = self.get_edge_logits(slg, in_trip_logits, out_trip_logits, temp, is_hard)  
#                         survived_ratios['lg'].append((w_triplets.detach().sum() / lg.num_edges()).item())
#                     else:
#                         w_triplets = torch.ones(slg.num_edges(), requires_grad=False).to(x.device)
#                 #------------------------------------------
#             else:
#                 """In a case of no action net, all edge weight is one."""
#                 w_bonds = torch.ones(g.num_edges(), requires_grad=False).to(x.device)
#                 w_triplets = torch.ones(lg.num_edges(), requires_grad=False).to(x.device)
                
#             # Policy net for node states on Line Graph.
#             # if keep_bond_forcing:
#             #     slg = batch_node_subgraph(lg, torch.ones(g.num_edges()).to(x.device))
#             # else:
#             #     slg = batch_node_subgraph(lg, w_bonds.bool())
                
#             # if self.num_act_linegraph_layers > 0:
#             #     triplet_mask = slg.edata[dgl.EID]
#             #     in_triplet_logits, _ = self.in_act_linegraph_net(slg, y[w_bonds.bool()], z[triplet_mask])  # (N, 2)
#             #     out_triplet_logits, _ = self.out_act_linegraph_net(slg, y[w_bonds.bool()], z[triplet_mask])  # (N, 2)

#             #     temp = self.temp_model(g=slg, edge_attr=z) if self.learn_temp else self.temp
#             #     w_triplets = self.get_edge_logits(slg, in_triplet_logits, out_triplet_logits, temp)

#             #     survived_ratios['lg'].append(w_triplets.detach().sum() / lg.num_edges()).item())
#             #     survived_ratios['slg'].append((w_triplets.detach().sum() / slg.num_edges()).item())
#             # else:
#             #     """In a case of no action net, all edge weight is one."""
#             #     w_triplets = torch.ones(slg.num_edges(), requires_grad=False).to(x.device)

#             # ALIGNN updates: update node, edge, triplet features
#             x, y, z = alignn_layer(g, lg, x, y, z, w_bonds, w_triplets, keep_bond_forcing)

#         # gated GCN updates: update node, edge features
#         for gcn_layer in self.gcn_layers:
#             # Policy net for node states on Graph.
#             if self.num_act_graph_layers > 0:
#                 x_act = x
#                 # x_act = self.act_atom_embedding(x)
#                 y_act = self.act_edge_embedding(y)
                
#                 in_gcn_logits, _ = self.in_act_graph_net(g, x_act, y_act)  # (N, 2)
#                 out_gcn_logits, _ = self.out_act_graph_net(g, x_act, y_act)  # (N, 2)

#                 temp = self.temp_model(g=g, edge_attr=y) if self.learn_temp else self.temp
#                 w_bonds = self.get_edge_logits(g, in_gcn_logits, out_gcn_logits, temp, is_hard)
#                 src, dst = g.edges()

#                 survived_ratios['gcn'].append((w_bonds.detach().sum() / g.num_edges()).item())
#             else:
#                 """In a case of no action net, all edge weight is one."""
#                 w_bonds = torch.ones(g.num_edges(), requires_grad=False).to(x.device)
                
#             x, y = gcn_layer(g, x, y, w_bonds)
#             if self.config.mask_before_next:
#                 y = y * w_bonds

#         # norm-activation-pool-classify
#         h = self.readout(g, x)
        
#         glvl_feats = [h]
#         if self.config.extra_features is not None:
#             h_feat = self.readout_feat(g, features)
#             glvl_feats.append(h_feat)

#         if self.config.global_features is not None:
#             for k, layer in self.global_feature_embedding.items():
#                 h_global = g.gdata.pop(k)
#                 h_global = h_global.to(x.device)
#                 h_global = layer(h_global)
#                 glvl_feats.append(h_global)

#         h = torch.cat(glvl_feats, dim=-1)        
#         for layer in self.fc:
#             h = layer(h)

#         if self.link:
#             h = self.link(h)

#         if self.classification:
#             # out = torch.round(torch.sigmoid(out))
#             h = self.softmax(h)
#         return torch.squeeze(h), survived_ratios, w_bonds, in_gcn_logits, out_gcn_logits

#     def create_edge_weight(self, edge_index: Tuple[torch.Tensor, torch.Tensor], 
#                            keep_in_prob: torch.Tensor, 
#                            keep_out_prob: torch.Tensor) -> torch.Tensor:
#         u, v = edge_index
#         edge_in_prob = keep_in_prob[v]
#         edge_out_prob = keep_out_prob[u]
#         return edge_in_prob * edge_out_prob

#     def get_edge_logits(self, g, in_logits, out_logits, temp, is_hard):        
#         in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=is_hard)
#         out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=is_hard)
#         w_edges = self.create_edge_weight(edge_index=g.edges(),
#                                           keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])
#         return w_edges


class CoALIGNN_inAction(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.config = config
        self.classification = config.classification

        input_node_dim = 0
        if config.atom_input_features > 0:
            self.atom_embedding = MLPLayer(
                config.atom_input_features, config.hidden_features
            )
            input_node_dim += config.hidden_features

        if config.extra_atom_features is not None:
            if isinstance(config.extra_atom_features, int):
                self.config.extra_atom_features = {"extra_atom_features": config.extra_atom_features}
            
            self.extra_atom_embedding = nn.ModuleDict()
            for k, v in self.config.extra_atom_features.items():
                self.extra_atom_embedding[k] = MLPLayer(
                    v, config.hidden_features
                )
                input_node_dim += config.hidden_features

            self.squeeze_atom_embedding = MLPLayer(
                    input_node_dim, config.hidden_features
                )
            input_node_dim = config.hidden_features

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, input_node_dim),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, input_node_dim),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    input_node_dim,
                    input_node_dim,
                    config.mask_before_next,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    input_node_dim, input_node_dim
                )
                for idx in range(config.gcn_layers)
            ]
        )

        self.is_hard = config.is_hard
        self.mask_triplets = config.mask_triplets
        self.num_act_graph_layers = config.action_args.num_act_graph_layers
        if self.num_act_graph_layers > 0:
            # self.act_atom_embedding = MLPLayer(
            #     input_node_dim, input_node_dim
            # )     
            # self.act_edge_embedding = nn.Sequential(
            #     RBFExpansion(
            #         vmin=0,
            #         vmax=8.0,
            #         bins=config.edge_input_features,
            #     ),
            #     MLPLayer(config.edge_input_features, config.embedding_features),
            #     MLPLayer(config.embedding_features, config.hidden_features),
            # )
            self.act_edge_embedding = MLPLayer(
                config.hidden_features, config.hidden_features
            )
            self.in_act_graph_net = ActionNet(self.num_act_graph_layers, 
                                              input_node_dim, 2)
            self.out_act_graph_net = ActionNet(self.num_act_graph_layers, 
                                               input_node_dim, 2)
            
        self.num_act_linegraph_layers = config.action_args.num_act_linegraph_layers
        if self.num_act_linegraph_layers > 0:
            self.in_act_linegraph_net = ActionNet(self.num_act_linegraph_layers, 
                                              input_features, 2)
            self.out_act_linegraph_net = ActionNet(self.num_act_graph_layers, 
                                               input_features, 2)

        self.learn_temp = config.learn_temp
        self.temp = config.temp

        self.readout = AvgPooling()
        self.readout_feat = AvgPooling()
        if self.classification:
            self.fc = nn.Linear(config.hidden_features, config.num_classes)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)

        # if config.extra_features > 0:
        #     # Credit for extra_features work:
        #     # Gong et al., https://doi.org/10.48550/arXiv.2208.05039
        #     self.extra_feature_embedding = MLPLayer(
        #         config.extra_features, config.extra_features
        #     )
        hidden_dim = input_node_dim
        if config.extra_features is not None:
            if isinstance(config.extra_features, int):
                self.config.extra_features = {"extra_features": config.extra_features}            
            self.extra_feature_embedding = nn.ModuleDict()
            for k, v in self.config.extra_features.items():
                self.extra_feature_embedding[k] = MLPLayer(v, v)  
                hidden_dim += v

        if config.global_features is not None:
            if isinstance(config.global_features, int):
                self.config.global_features = {"global_features": config.global_features}
            
            self.global_feature_embedding = nn.ModuleDict()
            for k, v in self.config.global_features.items():
                if config.embed_global_features:
                    self.global_feature_embedding[k] = nn.Sequential(
                        MLPLayer(v, config.hidden_features), 
                        # nn.Linear(config.hidden_features, config.hidden_features),
                        )                    
                    hidden_dim += config.hidden_features
                else:
                    self.global_feature_embedding[k] = nn.Identity()
                    hidden_dim += v
            
        self.fc = nn.ModuleList()
        if (config.extra_features is not None) or (config.global_features is not None):
            for _ in range(2):
                self.fc.append(MLPLayer(hidden_dim, hidden_dim))
        self.fc.append(nn.Linear(hidden_dim, config.output_features))

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph],
        keep_bond_forcing: bool=False, context=torch.enable_grad(), is_hard=False,
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if len(self.alignn_layers) > 0:
            g, lg = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))
        else:
            g, lg = g

        # In an updated alignn code, extra node-level features are 
        # embeded using mlp and then pooled to get graph-level features. 
        # before concatenating with the graph-level features pooled
        # from gnn.
        # if self.config.extra_features != 0:
        #     features = g.ndata["extra_features"]
        #     features = self.extra_feature_embedding(features)
        if self.config.extra_features is not None:
            features = []
            for k, layer in self.extra_feature_embedding.items():
                h_extra = g.ndata.pop(k)
                h_extra = layer(h_extra)
                features.append(h_extra)
            features = torch.cat(features, dim=-1)

        g = g.local_var()
        
        # initial node features: atom feature network...
        if self.config.atom_input_features > 0:
            x = g.ndata.pop("atom_features")
            x = self.atom_embedding(x)
        if self.config.extra_atom_features is not None:
            atom_feats = [x] if self.config.atom_input_features > 0 else []
            for k, layer in self.extra_atom_embedding.items():
                f = g.ndata.pop(k)
                atom_feats.append(layer(f))
            x = torch.cat(atom_feats, dim=-1)
            x = self.squeeze_atom_embedding(x)

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        survived_ratios = {'g': [], 'lg': [], 'slg': [], 'gcn': []}
        

                


        # gated GCN updates: update node, edge features
        in_probs = None
        for gcn_layer in self.gcn_layers:
            # Policy net for node states on Graph.
            if self.num_act_graph_layers > 0:
                x_act = x
                # x_act = self.act_atom_embedding(x)
                y_act = self.act_edge_embedding(y)
                
                in_gcn_logits, _ = self.in_act_graph_net(g, x_act, y_act)  # (N, 2)
                # out_gcn_logits, _ = self.out_act_graph_net(g, x_act, y_act)  # (N, 2)

                temp = self.temp_model(g=g, edge_attr=y) if self.learn_temp else self.temp
                in_probs = self.get_edge_gum_prob(g,in_gcn_logits,temp,is_hard)
                # w_bonds = self.get_edge_logits(g, in_gcn_logits, out_gcn_logits, temp, is_hard)
                src, dst = g.edges()

                # survived_ratios['gcn'].append((w_bonds.detach().sum() / g.num_edges()).item())
            else:
                """In a case of no action net, all edge weight is one."""
                # w_bonds = torch.ones(g.num_edges(), requires_grad=False).to(x.device)
                
            # x, y = gcn_layer(g, x, y, w_bonds)
            # if self.config.mask_before_next:
            #     y = y * w_bonds

        # norm-activation-pool-classify
        # h = self.readout(g, x)
        
        # glvl_feats = [h]
        # if self.config.extra_features is not None:
        #     h_feat = self.readout_feat(g, features)
        #     glvl_feats.append(h_feat)

        # if self.config.global_features is not None:
        #     for k, layer in self.global_feature_embedding.items():
        #         h_global = g.gdata.pop(k)
        #         h_global = h_global.to(x.device)
        #         h_global = layer(h_global)
        #         glvl_feats.append(h_global)

        # h = torch.cat(glvl_feats, dim=-1)        
        # for layer in self.fc:
        #     h = layer(h)

        # if self.link:
        #     h = self.link(h)

        # if self.classification:
        #     # out = torch.round(torch.sigmoid(out))
        #     h = self.softmax(h)
        return in_probs
        # return torch.squeeze(h), survived_ratios, w_bonds, in_gcn_logits, out_gcn_logits


    def create_edge_weight(self, edge_index: Tuple[torch.Tensor, torch.Tensor], 
                           keep_in_prob: torch.Tensor, 
                           keep_out_prob: torch.Tensor) -> torch.Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob

    def get_edge_logits(self, g, in_logits, out_logits, temp, is_hard):        
        in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=is_hard)
        out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=is_hard)
        w_edges = self.create_edge_weight(edge_index=g.edges(),
                                          keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])
        return w_edges

    def get_edge_weights(self, g, in_probs, out_probs):        

        w_edges = self.create_edge_weight(edge_index=g.edges(),
                                          keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])
        return w_edges

    def get_edge_gum_prob(self, g, logits, temp, is_hard):        
        probs = F.gumbel_softmax(logits=logits, tau=temp, hard=is_hard)
        # out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=is_hard)
        # w_edges = self.create_edge_weight(edge_index=g.edges(),
        #                                   keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])
        return probs


class CoALIGNN_outAction(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.config = config
        self.classification = config.classification

        input_node_dim = 0
        if config.atom_input_features > 0:
            self.atom_embedding = MLPLayer(
                config.atom_input_features, config.hidden_features
            )
            input_node_dim += config.hidden_features

        if config.extra_atom_features is not None:
            if isinstance(config.extra_atom_features, int):
                self.config.extra_atom_features = {"extra_atom_features": config.extra_atom_features}
            
            self.extra_atom_embedding = nn.ModuleDict()
            for k, v in self.config.extra_atom_features.items():
                self.extra_atom_embedding[k] = MLPLayer(
                    v, config.hidden_features
                )
                input_node_dim += config.hidden_features

            self.squeeze_atom_embedding = MLPLayer(
                    input_node_dim, config.hidden_features
                )
            input_node_dim = config.hidden_features

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, input_node_dim),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, input_node_dim),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    input_node_dim,
                    input_node_dim,
                    config.mask_before_next,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    input_node_dim, input_node_dim
                )
                for idx in range(config.gcn_layers)
            ]
        )

        self.is_hard = config.is_hard
        self.mask_triplets = config.mask_triplets
        self.num_act_graph_layers = config.action_args.num_act_graph_layers
        if self.num_act_graph_layers > 0:
            # self.act_atom_embedding = MLPLayer(
            #     input_node_dim, input_node_dim
            # )     
            # self.act_edge_embedding = nn.Sequential(
            #     RBFExpansion(
            #         vmin=0,
            #         vmax=8.0,
            #         bins=config.edge_input_features,
            #     ),
            #     MLPLayer(config.edge_input_features, config.embedding_features),
            #     MLPLayer(config.embedding_features, config.hidden_features),
            # )
            self.act_edge_embedding = MLPLayer(
                config.hidden_features, config.hidden_features
            )
            self.in_act_graph_net = ActionNet(self.num_act_graph_layers, 
                                              input_node_dim, 2)
            self.out_act_graph_net = ActionNet(self.num_act_graph_layers, 
                                               input_node_dim, 2)
            
        self.num_act_linegraph_layers = config.action_args.num_act_linegraph_layers
        if self.num_act_linegraph_layers > 0:
            self.in_act_linegraph_net = ActionNet(self.num_act_linegraph_layers, 
                                              input_features, 2)
            self.out_act_linegraph_net = ActionNet(self.num_act_graph_layers, 
                                               input_features, 2)

        self.learn_temp = config.learn_temp
        self.temp = config.temp

        self.readout = AvgPooling()
        self.readout_feat = AvgPooling()
        if self.classification:
            self.fc = nn.Linear(config.hidden_features, config.num_classes)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)

        # if config.extra_features > 0:
        #     # Credit for extra_features work:
        #     # Gong et al., https://doi.org/10.48550/arXiv.2208.05039
        #     self.extra_feature_embedding = MLPLayer(
        #         config.extra_features, config.extra_features
        #     )
        hidden_dim = input_node_dim
        if config.extra_features is not None:
            if isinstance(config.extra_features, int):
                self.config.extra_features = {"extra_features": config.extra_features}            
            self.extra_feature_embedding = nn.ModuleDict()
            for k, v in self.config.extra_features.items():
                self.extra_feature_embedding[k] = MLPLayer(v, v)  
                hidden_dim += v

        if config.global_features is not None:
            if isinstance(config.global_features, int):
                self.config.global_features = {"global_features": config.global_features}
            
            self.global_feature_embedding = nn.ModuleDict()
            for k, v in self.config.global_features.items():
                if config.embed_global_features:
                    self.global_feature_embedding[k] = nn.Sequential(
                        MLPLayer(v, config.hidden_features), 
                        # nn.Linear(config.hidden_features, config.hidden_features),
                        )                    
                    hidden_dim += config.hidden_features
                else:
                    self.global_feature_embedding[k] = nn.Identity()
                    hidden_dim += v
            
        self.fc = nn.ModuleList()
        if (config.extra_features is not None) or (config.global_features is not None):
            for _ in range(2):
                self.fc.append(MLPLayer(hidden_dim, hidden_dim))
        self.fc.append(nn.Linear(hidden_dim, config.output_features))

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph],
        keep_bond_forcing: bool=False, context=torch.enable_grad(), is_hard=False,
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if len(self.alignn_layers) > 0:
            g, lg = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))
        else:
            g, lg = g

        # In an updated alignn code, extra node-level features are 
        # embeded using mlp and then pooled to get graph-level features. 
        # before concatenating with the graph-level features pooled
        # from gnn.
        # if self.config.extra_features != 0:
        #     features = g.ndata["extra_features"]
        #     features = self.extra_feature_embedding(features)
        if self.config.extra_features is not None:
            features = []
            for k, layer in self.extra_feature_embedding.items():
                h_extra = g.ndata.pop(k)
                h_extra = layer(h_extra)
                features.append(h_extra)
            features = torch.cat(features, dim=-1)

        g = g.local_var()
        
        # initial node features: atom feature network...
        if self.config.atom_input_features > 0:
            x = g.ndata.pop("atom_features")
            x = self.atom_embedding(x)
        if self.config.extra_atom_features is not None:
            atom_feats = [x] if self.config.atom_input_features > 0 else []
            for k, layer in self.extra_atom_embedding.items():
                f = g.ndata.pop(k)
                atom_feats.append(layer(f))
            x = torch.cat(atom_feats, dim=-1)
            x = self.squeeze_atom_embedding(x)

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        survived_ratios = {'g': [], 'lg': [], 'slg': [], 'gcn': []}
        

                


        # gated GCN updates: update node, edge features
        out_probs = None
        for gcn_layer in self.gcn_layers:
            # Policy net for node states on Graph.
            if self.num_act_graph_layers > 0:
                x_act = x
                # x_act = self.act_atom_embedding(x)
                y_act = self.act_edge_embedding(y)
                
                # in_gcn_logits, _ = self.in_act_graph_net(g, x_act, y_act)  # (N, 2)
                out_gcn_logits, _ = self.out_act_graph_net(g, x_act, y_act)  # (N, 2)

                temp = self.temp_model(g=g, edge_attr=y) if self.learn_temp else self.temp
                out_probs = self.get_edge_gum_prob(g,out_gcn_logits,temp,is_hard)
                # w_bonds = self.get_edge_logits(g, in_gcn_logits, out_gcn_logits, temp, is_hard)
                src, dst = g.edges()

                # survived_ratios['gcn'].append((w_bonds.detach().sum() / g.num_edges()).item())
            else:
                """In a case of no action net, all edge weight is one."""
                # w_bonds = torch.ones(g.num_edges(), requires_grad=False).to(x.device)
                
            # x, y = gcn_layer(g, x, y, w_bonds)
            # if self.config.mask_before_next:
            #     y = y * w_bonds

        # norm-activation-pool-classify
        # h = self.readout(g, x)
        
        # glvl_feats = [h]
        # if self.config.extra_features is not None:
        #     h_feat = self.readout_feat(g, features)
        #     glvl_feats.append(h_feat)

        # if self.config.global_features is not None:
        #     for k, layer in self.global_feature_embedding.items():
        #         h_global = g.gdata.pop(k)
        #         h_global = h_global.to(x.device)
        #         h_global = layer(h_global)
        #         glvl_feats.append(h_global)

        # h = torch.cat(glvl_feats, dim=-1)        
        # for layer in self.fc:
        #     h = layer(h)

        # if self.link:
        #     h = self.link(h)

        # if self.classification:
        #     # out = torch.round(torch.sigmoid(out))
        #     h = self.softmax(h)
        return out_probs
        # return torch.squeeze(h), survived_ratios, w_bonds, in_gcn_logits, out_gcn_logits


    def create_edge_weight(self, edge_index: Tuple[torch.Tensor, torch.Tensor], 
                           keep_in_prob: torch.Tensor, 
                           keep_out_prob: torch.Tensor) -> torch.Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob

    def get_edge_logits(self, g, in_logits, out_logits, temp, is_hard):        
        in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=is_hard)
        out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=is_hard)
        w_edges = self.create_edge_weight(edge_index=g.edges(),
                                          keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])
        return w_edges

    def get_edge_weights(self, g, in_probs, out_probs):        

        w_edges = self.create_edge_weight(edge_index=g.edges(),
                                          keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])
        return w_edges

    def get_edge_gum_prob(self, g, logits, temp, is_hard):        
        probs = F.gumbel_softmax(logits=logits, tau=temp, hard=is_hard)
        # out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=is_hard)
        # w_edges = self.create_edge_weight(edge_index=g.edges(),
        #                                   keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])
        return probs

class CoALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.config = config
        self.classification = config.classification

        input_node_dim = 0
        if config.atom_input_features > 0:
            self.atom_embedding = MLPLayer(
                config.atom_input_features, config.hidden_features
            )
            input_node_dim += config.hidden_features

        if config.extra_atom_features is not None:
            if isinstance(config.extra_atom_features, int):
                self.config.extra_atom_features = {"extra_atom_features": config.extra_atom_features}
            
            self.extra_atom_embedding = nn.ModuleDict()
            for k, v in self.config.extra_atom_features.items():
                self.extra_atom_embedding[k] = MLPLayer(
                    v, config.hidden_features
                )
                input_node_dim += config.hidden_features

            self.squeeze_atom_embedding = MLPLayer(
                    input_node_dim, config.hidden_features
                )
            input_node_dim = config.hidden_features

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, input_node_dim),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, input_node_dim),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    input_node_dim,
                    input_node_dim,
                    config.mask_before_next,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    input_node_dim, input_node_dim
                )
                for idx in range(config.gcn_layers)
            ]
        )

        self.is_hard = config.is_hard
        self.mask_triplets = config.mask_triplets
        self.num_act_graph_layers = config.action_args.num_act_graph_layers
        if self.num_act_graph_layers > 0:
            # self.act_atom_embedding = MLPLayer(
            #     input_node_dim, input_node_dim
            # )     
            # self.act_edge_embedding = nn.Sequential(
            #     RBFExpansion(
            #         vmin=0,
            #         vmax=8.0,
            #         bins=config.edge_input_features,
            #     ),
            #     MLPLayer(config.edge_input_features, config.embedding_features),
            #     MLPLayer(config.embedding_features, config.hidden_features),
            # )
            self.act_edge_embedding = MLPLayer(
                config.hidden_features, config.hidden_features
            )
            self.in_act_graph_net = ActionNet(self.num_act_graph_layers, 
                                              input_node_dim, 2)
            self.out_act_graph_net = ActionNet(self.num_act_graph_layers, 
                                               input_node_dim, 2)
            
        self.num_act_linegraph_layers = config.action_args.num_act_linegraph_layers
        if self.num_act_linegraph_layers > 0:
            self.in_act_linegraph_net = ActionNet(self.num_act_linegraph_layers, 
                                              input_features, 2)
            self.out_act_linegraph_net = ActionNet(self.num_act_graph_layers, 
                                               input_features, 2)

        self.learn_temp = config.learn_temp
        self.temp = config.temp

        self.readout = AvgPooling()
        self.readout_feat = AvgPooling()
        if self.classification:
            self.fc = nn.Linear(config.hidden_features, config.num_classes)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)

        # if config.extra_features > 0:
        #     # Credit for extra_features work:
        #     # Gong et al., https://doi.org/10.48550/arXiv.2208.05039
        #     self.extra_feature_embedding = MLPLayer(
        #         config.extra_features, config.extra_features
        #     )
        hidden_dim = input_node_dim
        if config.extra_features is not None:
            if isinstance(config.extra_features, int):
                self.config.extra_features = {"extra_features": config.extra_features}            
            self.extra_feature_embedding = nn.ModuleDict()
            for k, v in self.config.extra_features.items():
                self.extra_feature_embedding[k] = MLPLayer(v, v)  
                hidden_dim += v

        if config.global_features is not None:
            if isinstance(config.global_features, int):
                self.config.global_features = {"global_features": config.global_features}
            
            self.global_feature_embedding = nn.ModuleDict()
            for k, v in self.config.global_features.items():
                if config.embed_global_features:
                    self.global_feature_embedding[k] = nn.Sequential(
                        MLPLayer(v, config.hidden_features), 
                        # nn.Linear(config.hidden_features, config.hidden_features),
                        )                    
                    hidden_dim += config.hidden_features
                else:
                    self.global_feature_embedding[k] = nn.Identity()
                    hidden_dim += v
            
        self.fc = nn.ModuleList()
        if (config.extra_features is not None) or (config.global_features is not None):
            for _ in range(2):
                self.fc.append(MLPLayer(hidden_dim, hidden_dim))
        self.fc.append(nn.Linear(hidden_dim, config.output_features))

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph],in_probs,out_probs,
        keep_bond_forcing: bool=False, context=torch.enable_grad(), is_hard=False,
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if len(self.alignn_layers) > 0:
            g, lg = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))
        else:
            g, lg = g

        # In an updated alignn code, extra node-level features are 
        # embeded using mlp and then pooled to get graph-level features. 
        # before concatenating with the graph-level features pooled
        # from gnn.
        # if self.config.extra_features != 0:
        #     features = g.ndata["extra_features"]
        #     features = self.extra_feature_embedding(features)
        if self.config.extra_features is not None:
            features = []
            for k, layer in self.extra_feature_embedding.items():
                h_extra = g.ndata.pop(k)
                h_extra = layer(h_extra)
                features.append(h_extra)
            features = torch.cat(features, dim=-1)

        g = g.local_var()
        
        # initial node features: atom feature network...
        if self.config.atom_input_features > 0:
            x = g.ndata.pop("atom_features")
            x = self.atom_embedding(x)
        if self.config.extra_atom_features is not None:
            atom_feats = [x] if self.config.atom_input_features > 0 else []
            for k, layer in self.extra_atom_embedding.items():
                f = g.ndata.pop(k)
                atom_feats.append(layer(f))
            x = torch.cat(atom_feats, dim=-1)
            x = self.squeeze_atom_embedding(x)

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        survived_ratios = {'g': [], 'lg': [], 'slg': [], 'gcn': []}
        

                


        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            # Policy net for node states on Graph.
            # if self.num_act_graph_layers > 0:
            #     x_act = x
            #     # x_act = self.act_atom_embedding(x)
            #     y_act = self.act_edge_embedding(y)
                
            #     # in_gcn_logits, _ = self.in_act_graph_net(g, x_act, y_act)  # (N, 2)
            #     out_gcn_logits, _ = self.out_act_graph_net(g, x_act, y_act)  # (N, 2)

            #     temp = self.temp_model(g=g, edge_attr=y) if self.learn_temp else self.temp
            #     out_probs = self.get_edge_gum_prob(g,out_gcn_logits,temp,is_hard)
            if in_probs is not None and out_probs is not None:
                w_bonds = self.get_edge_weights(g, in_probs, out_probs)
            else:
                w_bonds = torch.ones(g.num_edges(), requires_grad=False).to(x.device)

            if self.config.random_weight == 1:
                
                w_bonds = torch.rand(g.num_edges(), requires_grad=False).to(x.device)



            #     src, dst = g.edges()
            # print('w_bonds sum',w_bonds.detach().sum())
            # print(w_bonds)
            # print('g number of edges',g.num_edges())

            survived_ratios['gcn'].append((w_bonds.detach().sum() / g.num_edges()).item())
            # else:
            #     """In a case of no action net, all edge weight is one."""
                # w_bonds = torch.ones(g.num_edges(), requires_grad=False).to(x.device)
                
            x, y = gcn_layer(g, x, y, w_bonds)
            # x, y = gcn_layer(g, x, y)

            if self.config.mask_before_next:
                y = y * w_bonds

        # norm-activation-pool-classify
        h = self.readout(g, x)
        
        glvl_feats = [h]
        if self.config.extra_features is not None:
            h_feat = self.readout_feat(g, features)
            glvl_feats.append(h_feat)

        if self.config.global_features is not None:
            for k, layer in self.global_feature_embedding.items():
                h_global = g.gdata.pop(k)
                h_global = h_global.to(x.device)
                h_global = layer(h_global)
                glvl_feats.append(h_global)

        h = torch.cat(glvl_feats, dim=-1)        
        for layer in self.fc:
            h = layer(h)

        if self.link:
            h = self.link(h)

        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            h = self.softmax(h)
        # return out_probs
        return torch.squeeze(h), survived_ratios, w_bonds, 

    def create_edge_weight(self, edge_index: Tuple[torch.Tensor, torch.Tensor], 
                           keep_in_prob: torch.Tensor, 
                           keep_out_prob: torch.Tensor) -> torch.Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob

    def get_edge_logits(self, g, in_logits, out_logits, temp, is_hard):        
        in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=is_hard)
        out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=is_hard)
        w_edges = self.create_edge_weight(edge_index=g.edges(),
                                          keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])
        return w_edges

    def get_edge_weights(self, g, in_probs, out_probs):        

        w_edges = self.create_edge_weight(edge_index=g.edges(),
                                          keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])
        return w_edges

    def get_edge_gum_prob(self, g, logits, temp, is_hard):        
        probs = F.gumbel_softmax(logits=logits, tau=temp, hard=is_hard)
        # out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=is_hard)
        # w_edges = self.create_edge_weight(edge_index=g.edges(),
        #                                   keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])
        return probs


class ActionNet(nn.Module):
    def __init__(self, num_layers, hidden_dim, out_dim):
        """
        Create a model which represents the agent's policy.
        """
        super().__init__()
        self.num_layers = num_layers
        self.nets = nn.ModuleList(
            [
                EdgeGatedGraphConv(hidden_dim, hidden_dim)
                for idx in range(num_layers-1)
            ] 
        ) 
        self.nets.append(EdgeGatedGraphConv(hidden_dim, 
                                            out_dim, 
                                            residual=False))

    def forward(self, g: dgl.DGLGraph, 
                x: torch.Tensor, 
                y: torch.Tensor) -> torch.Tensor:
        for layer in self.nets:
            x, y = layer(g, x, y)
        return x, y
