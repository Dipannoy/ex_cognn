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
import dgl.nn as dglnn

#from alignn.models.utils import RBFExpansion
#from alignn.utils import BaseSettings

from .utils import batch_node_subgraph, batch_edge_subgraph, RBFExpansion
#from .layers import WeightedGINEConv
#from .action import ActionNetForTest
import dgl.function as fn



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


class CGCNNConv(nn.Module):
    """CGCNN-style graph convolution from arxiv:1710.10324."""
    
    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for CGCNN update."""
        super().__init__()
        self.residual = residual
        
        # In CGCNN, edge features are concatenated with node features
        # So we need a linear layer that takes (node_feat + node_feat + edge_feat)
        self.linear = nn.Linear(2 * input_features + input_features, output_features)
        
        # Batch norms
        self.bn_nodes = nn.BatchNorm1d(output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)
        
        # Edge update function (optional)
        self.edge_update = nn.Linear(input_features, output_features)
        
    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_weights: torch.Tensor = None,
        node_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """CGCNN graph convolution.
        
        h_i^l+1 = ReLU(W [h_i || h_j || e_ij] + b)
        """
        g = g.local_var()
        
        # Store node features
        g.ndata['h'] = node_feats
        
        # Compute messages: concatenate src, dst, edge features
        # g.apply_edges(lambda edges: {'m': torch.cat([
        #     edges.src['h'], 
        #     edges.dst['h'], 
        #     edge_feats[edges.edata[dgl.EID]]
        # , dim=-1)})
        
        g.apply_edges(lambda edges: {'m': torch.cat([
            edges.src['h'], 
            edges.dst['h'], 
            edge_feats[edges.edata[dgl.EID]]
        ], dim=-1)
        })
        
        # Apply linear transformation to messages
        m = self.linear(g.edata.pop('m'))
        
        # Create a subgraph if node_mask is provided
        if node_mask is not None:
            g = batch_node_subgraph(g, node_mask)
            edge_mask = g.edata[dgl.EID]
        else:
            node_mask = torch.ones((node_feats.shape[0],)).bool().to(m.device)
            edge_mask = torch.ones((edge_feats.shape[0],)).bool().to(m.device)
        
        # Apply edge weights if provided
        edge_weights = edge_weights.view(-1,1) if edge_weights is not None else 1.
        g.edata['m'] = edge_weights * m[edge_mask]
        
        # Sum messages to update nodes
        g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h_new'))
        
        # Get updated node features
        x = torch.zeros_like(node_feats)
        x[node_mask] = g.ndata.pop('h_new')
        
        # Update edge features (optional)
        y = self.edge_update(edge_feats)
        
        # Apply activation and batch norm
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(y))
        
        if self.residual:
            x = node_feats + x
            y = edge_feats + y
            
        return x, y


class MeanGNNWithEdgeFeatures(nn.Module):
    def __init__(self, hidden_feats, out_feats):
        super(MeanGNNWithEdgeFeatures, self).__init__()
        self.node_linear = nn.Linear(hidden_feats, hidden_feats)  # Transform node features
        self.edge_linear = nn.Linear(hidden_feats, hidden_feats)  # Transform edge features
        self.final_linear = nn.Linear(hidden_feats, out_feats)     # Final transformation
    
    def forward(self, g, x, y):
        # Message-passing with custom edge features
        def message_func(edges):
            # Combine edge features and source node features
            return {'msg': edges.src['h'] + edges.data['he']}
        
        def reduce_func(nodes):
            # Aggregate messages (mean aggregation)
            return {'h': torch.mean(nodes.mailbox['msg'], dim=1)}
        
        # Initialize node and edge representations
        g.ndata['h'] = self.node_linear(x)
        g.edata['he'] = self.edge_linear(y)
        
        # Perform message passing
        g.update_all(message_func, reduce_func)
        
        # Apply final transformation
        h = g.ndata.pop('h')
        return F.log_softmax(self.final_linear(h), dim=1)


class MeanGNN(nn.Module):
    def __init__(self, hidden_feats, out_feats):
        super(MeanGNN, self).__init__()
        # self.conv1 = dgl.nn.GraphConv(hidden_feats, hidden_feats, aggregate='mean')
        # self.conv2 = dgl.nn.GraphConv(hidden_feats, out_feats, aggregate='mean')

        self.conv1 = dglnn.SAGEConv(
            in_feats=hidden_feats,
            out_feats=hidden_feats,
            aggregator_type='mean'  # Options: 'mean', 'pool', 'lstm', 'gcn'
        )

        self.conv2 = dglnn.SAGEConv(
            in_feats=hidden_feats,
            out_feats=out_feats,
            aggregator_type='mean'  # Options: 'mean', 'pool', 'lstm', 'gcn'
        )

    def forward(self, g, features):
        # First layer
        h = self.conv1(g, features)
        h = F.relu(h)
        
        # Second layer
        h = self.conv2(g, h)
        
        return F.log_softmax(h, dim=1)


class WeightedGNNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super(WeightedGNNConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Linear transformation for concatenated features
        self.lin = nn.Linear(2 * in_channels, out_channels, bias=bias)

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, edge_attr: torch.Tensor = None, edge_weight: torch.Tensor = None):
        """
        Forward method for DGL implementation of WeightedGNNConv.

        Args:
            g (dgl.DGLGraph): Input graph.
            x (torch.Tensor): Node features of shape [num_nodes, in_channels].
            edge_weight (torch.Tensor, optional): Edge weights of shape [num_edges].
            edge_attr (torch.Tensor, optional): Edge attributes of shape [num_edges, *].

        Returns:
            torch.Tensor: Updated node features of shape [num_nodes, out_channels].
        """
        # Save node features in the graph
        g.ndata['x'] = x

        # Optional: Combine edge weights and edge attributes
        if edge_weight is not None and edge_attr is not None:
            g.edata['weight'] = edge_weight.unsqueeze(-1) * edge_attr  # Element-wise product
        elif edge_weight is not None:
            g.edata['weight'] = edge_weight.unsqueeze(-1)  # Only edge weights
        elif edge_attr is not None:
            g.edata['weight'] = edge_attr  # Only edge attributes
        else:
            g.edata['weight'] = torch.ones((g.num_edges(), 1), device=x.device)  # Default weight = 1

        # Define message and reduce functions
        def message_func(edges):
            return {'msg': edges.src['x'] * edges.data['weight']}  # Weighted node features

        def reduce_func(nodes):
            return {'agg_msg': torch.mean(nodes.mailbox['msg'], dim=1)}  # Mean aggregation

        # Apply message passing
        g.update_all(message_func, reduce_func)

        # Retrieve the aggregated messages
        agg_msg = g.ndata.pop('agg_msg')

        # Concatenate input features and aggregated messages
        out = torch.cat([x, agg_msg], dim=-1)

        # Apply linear transformation
        out = self.lin(out)
        return out

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
        
# 	self.gcn_layers = nn.ModuleList(
#             [
#              	CGCNNConv(
#                     input_node_dim, input_node_dim
#                 )
#                 for idx in range(config.gcn_layers)
#             ]
# 	)

        self.is_hard = config.is_hard
        self.mask_triplets = config.mask_triplets
        self.edge_threshold = config.edge_threshold
        self.check_threshold = config.check_threshold
        self.no_edge = config.no_edge
        self.random_pruning = config.random_pruning
        self.random_drop_rate = config.random_drop_rate
        
        self.num_act_graph_layers = config.action_args.num_act_graph_layers
        # self.num_in_act_graph_layers = config.action_args.num_in_act_graph_layers
        self.num_in_act_graph_layers = 1

        self.action_net = config.action_args.action_net

    
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
            '''
            self.in_act_graph_net = ActionNet(self.num_act_graph_layers, 
                                              input_node_dim, 2)
            self.out_act_graph_net = ActionNet(self.num_act_graph_layers, 
                                               input_node_dim, 2)
            '''
            # if(self.action_net == 'ActionNet'):
            #     self.in_act_graph_net = ActionNet(self.num_in_act_graph_layers, 
            #                                     input_node_dim, 2)
            #     self.out_act_graph_net = ActionNet(self.num_act_graph_layers, 
            #                                     input_node_dim, 2)
            # elif(self.action_net == 'ActionNet_mean_gnn_node_edge'):
            #     self.in_act_graph_net = ActionNet_mean_gnn_node_edge(self.num_in_act_graph_layers, 
            #                                     input_node_dim, 2)
            #     self.out_act_graph_net = ActionNet_mean_gnn_node_edge(self.num_act_graph_layers, 
            #                                     input_node_dim, 2)
            # elif(self.action_net == 'ActionNet_mean_gnn_node'):
            #     self.in_act_graph_net = ActionNet_mean_gnn_node(self.num_in_act_graph_layers, 
            #                                     input_node_dim, 2)
            #     self.out_act_graph_net = ActionNet_mean_gnn_node(self.num_act_graph_layers, 
            #                                     input_node_dim, 2)
            if(self.action_net == 'EdgeScoringNet'):
                self.edge_scoring_net = EdgeScoringNet(
                    node_dim=input_node_dim,
                    edge_dim=input_node_dim,
                    hidden_dim=input_node_dim,
                    num_layers=max(self.num_act_graph_layers, 1),
                )
            elif(self.action_net == 'EdgeOnlyScoringNet'):
                self.edge_scoring_net = EdgeOnlyScoringNet(
                    node_dim=input_node_dim,
                    edge_dim=input_node_dim,
                    hidden_dim=input_node_dim,
                    num_layers=max(self.num_act_graph_layers, 1),
                )
            elif(self.action_net == 'LearnableCutoffScoringNet'):
                self.edge_scoring_net = LearnableCutoffScoringNet(
                    node_dim=input_node_dim,
                    edge_dim=input_node_dim,
                    hidden_dim=input_node_dim,
                    num_layers=max(self.num_act_graph_layers, 1),
                    init_r_cut=getattr(config, 'init_r_cut', 3.5),
                )
            elif(self.action_net == 'MultiplicativeGateScoringNet'):
                self.edge_scoring_net = MultiplicativeGateScoringNet(
                    node_dim=input_node_dim,
                    edge_dim=input_node_dim,
                    hidden_dim=input_node_dim,
                    num_layers=max(self.num_act_graph_layers, 1),
                )
            else:
                self.in_act_graph_net = ActionNet_mean_gnn_weighted(self.num_in_act_graph_layers, 
                                                input_node_dim, 2)
                self.out_act_graph_net = ActionNet_mean_gnn_weighted(self.num_act_graph_layers, 
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
        keep_bond_forcing: bool=False, context=torch.enable_grad(), is_hard=False,is_eval=False,
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
        # store raw bond lengths for LearnableCutoffScoringNet
        g.edata['bond_lengths'] = bondlength

        survived_ratios = {'g': [], 'lg': [], 'slg': [], 'gcn': []}
        
        for alignn_layer in self.alignn_layers:
            # Policy net for node states on Graph.
            if self.num_act_graph_layers > 0:
                x_act = x
                # x_act = self.act_atom_embedding(x)
                y_act = self.act_edge_embedding(y)

                with context:
                    in_bond_logits, in_trip_logits = self.in_act_graph_net(g, x_act, y_act)  # (N, 2)
                    out_bond_logits, out_trip_logits = self.out_act_graph_net(g, x_act, y_act)  # (N, 2)
    
                    temp = self.temp_model(g=g, edge_attr=y) if self.learn_temp else self.temp
                    w_bonds = self.get_edge_logits(g, in_bond_logits, out_bond_logits, temp, is_hard)
                    if(self.check_threshold==True):
                        w_bonds = w_bonds * (w_bonds >= self.edge_threshold).float()
                    

                    survived_ratios['g'].append((w_bonds.detach().sum() / g.num_edges()).item())
    
                    # not sure which one is better.
                    if self.mask_triplets:
                        # Policy net for node states on Line Graph.
                        if keep_bond_forcing:
                            slg = batch_node_subgraph(lg, torch.ones(g.num_edges()).to(x.device))
                        else:
                            slg = batch_node_subgraph(lg, w_bonds.bool())
                        w_triplets = self.get_edge_logits(slg, in_trip_logits, out_trip_logits, temp, is_hard)  
                        survived_ratios['lg'].append((w_triplets.detach().sum() / lg.num_edges()).item())
                    else:
                        w_triplets = torch.ones(slg.num_edges(), requires_grad=False).to(x.device)
                #------------------------------------------
            else:
                """In a case of no action net, all edge weight is one."""
                w_bonds = torch.ones(g.num_edges(), requires_grad=False).to(x.device)
                w_triplets = torch.ones(lg.num_edges(), requires_grad=False).to(x.device)
                
            # Policy net for node states on Line Graph.
            # if keep_bond_forcing:
            #     slg = batch_node_subgraph(lg, torch.ones(g.num_edges()).to(x.device))
            # else:
            #     slg = batch_node_subgraph(lg, w_bonds.bool())
                
            # if self.num_act_linegraph_layers > 0:
            #     triplet_mask = slg.edata[dgl.EID]
            #     in_triplet_logits, _ = self.in_act_linegraph_net(slg, y[w_bonds.bool()], z[triplet_mask])  # (N, 2)
            #     out_triplet_logits, _ = self.out_act_linegraph_net(slg, y[w_bonds.bool()], z[triplet_mask])  # (N, 2)

            #     temp = self.temp_model(g=slg, edge_attr=z) if self.learn_temp else self.temp
            #     w_triplets = self.get_edge_logits(slg, in_triplet_logits, out_triplet_logits, temp)

            #     survived_ratios['lg'].append(w_triplets.detach().sum() / lg.num_edges()).item())
            #     survived_ratios['slg'].append((w_triplets.detach().sum() / slg.num_edges()).item())
            # else:
            #     """In a case of no action net, all edge weight is one."""
            #     w_triplets = torch.ones(slg.num_edges(), requires_grad=False).to(x.device)

            # ALIGNN updates: update node, edge, triplet features
            x, y, z = alignn_layer(g, lg, x, y, z, w_bonds, w_triplets, keep_bond_forcing)

        # gated GCN updates: update node, edge features
        in_gcn_logits = None
        out_gcn_logits = None


        for gcn_layer in self.gcn_layers:
            if self.num_act_graph_layers > 0:
                x_act = x
                temp = self.temp_model(g=g, edge_attr=y) if self.learn_temp else self.temp

                if self.action_net in ('EdgeScoringNet', 'EdgeOnlyScoringNet',
                                       'LearnableCutoffScoringNet', 'MultiplicativeGateScoringNet'):
                    edge_logits = self.edge_scoring_net(g, x_act, y)  # [E, 2]
                    w_bonds = F.gumbel_softmax(logits=edge_logits, tau=temp, hard=is_hard)[:, 0]
                    in_gcn_logits = edge_logits   # store for downstream (shape [E,2], not [N,2])
                    out_gcn_logits = None
                elif self.action_net == 'ActionNet_mean_gnn_node':
                    y_act = self.act_edge_embedding(y)
                    in_gcn_logits = self.in_act_graph_net(g, x_act)   # [N, 2]
                    out_gcn_logits = self.out_act_graph_net(g, x_act)  # [N, 2]
                    w_bonds = self.get_edge_logits(g, in_gcn_logits, out_gcn_logits, temp, is_hard)
                else:
                    y_act = self.act_edge_embedding(y)
                    in_gcn_logits, _ = self.in_act_graph_net(g, x_act, y_act)   # [N, 2]
                    out_gcn_logits, _ = self.out_act_graph_net(g, x_act, y_act)  # [N, 2]
                    w_bonds = self.get_edge_logits(g, in_gcn_logits, out_gcn_logits, temp, is_hard)

                if self.check_threshold:
                    w_bonds = w_bonds * (w_bonds >= self.edge_threshold).float()
                survived_ratios['gcn'].append((w_bonds.detach().sum() / g.num_edges()).item())
            else:
                if self.no_edge:
                    w_bonds = torch.zeros(g.num_edges(), requires_grad=False).to(x.device)
                else:
                    w_bonds = torch.ones(g.num_edges(), requires_grad=False).to(x.device)
                    if self.random_pruning:
                        mask = torch.bernoulli(torch.full_like(w_bonds, 1-self.random_drop_rate)).to(x.device)
                        w_bonds = w_bonds * mask


            x, y = gcn_layer(g, x, y, w_bonds)
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
        return torch.squeeze(h), survived_ratios, w_bonds,in_gcn_logits,out_gcn_logits

    '''
    def create_edge_weight(self, edge_index: Tuple[torch.Tensor, torch.Tensor],
                           keep_in_prob: torch.Tensor) -> torch.Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        #edge_out_prob = keep_out_prob[u]
        return edge_in_prob 

    '''
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
    
    '''
    def get_edge_logits(self, g, in_logits, temp, is_hard):
        in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=is_hard)
        #out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=is_hard)
    
        #w_edges = self.create_edge_weight(edge_index=g.edges(),
        #                                  keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])

        w_edges = self.create_edge_weight(edge_index=g.edges(),
                                          keep_in_prob=in_probs[:, 0])

        return w_edges
    '''




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
'''
class ActionNet(nn.Module):
    def __init__(self, num_layers, hidden_dim, out_dim):
        """
	Create a model which represents the agent's policy.
        """
	super().__init__()
        self.num_layers = num_layers
        self.nets = nn.ModuleList(
            [
             	CGCNNConv(hidden_dim, hidden_dim)
                for idx in range(num_layers-1)
            ]
	)
	self.nets.append(CGCNNConv(hidden_dim,
                                   out_dim,
                                   residual=False))

    def forward(self, g: dgl.DGLGraph,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        for layer in self.nets:
            x, y = layer(g, x, y)
	return x, y
'''


class ActionNet_mean_gnn_node_edge(nn.Module):
    def __init__(self, num_layers, hidden_dim, out_dim):
        """
        Create a model which represents the agent's policy.
        """
        super().__init__()
        self.num_layers = num_layers
        self.nets = nn.ModuleList(
            [
                MeanGNNWithEdgeFeatures(hidden_dim, hidden_dim)
                for idx in range(num_layers-1)
            ]
        ) 
        self.nets.append(MeanGNNWithEdgeFeatures(hidden_dim, 
                                            out_dim))

    def forward(self, g: dgl.DGLGraph, 
                x: torch.Tensor, 
                y: torch.Tensor) -> torch.Tensor:
        for layer in self.nets:
            x = layer(g, x, y)
        return x


class ActionNet_mean_gnn_node(nn.Module):
    def __init__(self, num_layers, hidden_dim, out_dim):
        """
        Create a model which represents the agent's policy.
        """
        super().__init__()
        self.num_layers = num_layers
        self.nets = nn.ModuleList(
            [
                MeanGNN(hidden_dim, hidden_dim)
                for idx in range(num_layers-1)
            ]
        ) 
        self.nets.append(MeanGNN(hidden_dim, 
                                            out_dim))

    def forward(self, g: dgl.DGLGraph, 
                x: torch.Tensor) -> torch.Tensor:
        for layer in self.nets:
            x = layer(g, x)
        return x



class ActionNet_mean_gnn_weighted(nn.Module):
    def __init__(self, num_layers, hidden_dim, out_dim):
        """
        Create a model which represents the agent's policy.
        """
        super().__init__()
        self.num_layers = num_layers
        self.nets = nn.ModuleList(
            [
                WeightedGNNConv(hidden_dim, hidden_dim)
                for idx in range(num_layers-1)
            ]
        ) 
        self.nets.append(WeightedGNNConv(hidden_dim, 
                                            out_dim))

    def forward(self, g: dgl.DGLGraph, 
                x: torch.Tensor, 
                y: torch.Tensor) -> torch.Tensor:
        for layer in self.nets:
            x = layer(g, x, y)
        return x


class EdgeScoringNet(nn.Module):
    """Direct per-edge scoring from concatenated [h_src || h_dst || e_ij].

    Unlike node-level action nets, bond geometry (encoded in edge features via
    RBFExpansion) flows directly into the keep/drop decision, so the learned
    edge weights are geometrically grounded and correlated with bond length.

    Args:
        node_dim:   dimension of node features
        edge_dim:   dimension of edge features
        hidden_dim: hidden layer width
        num_layers: total MLP depth (including the final Linear → 2 layer)
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        in_dim = 2 * node_dim + edge_dim
        layers: list = []
        dim = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU()]
            dim = hidden_dim
        layers.append(nn.Linear(dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return per-edge logits of shape [E, 2]."""
        src, dst = g.edges()
        edge_input = torch.cat([x[src], x[dst], y], dim=-1)
        return self.net(edge_input)


class EdgeOnlyScoringNet(nn.Module):
    """Ablation: per-edge scorer that ignores node features entirely.

    Input is the edge embedding `e_ij` only (RBF-encoded bond length →
    hidden_dim via edge_embedding).  If this variant produces better
    bond-length separation than EdgeScoringNet it confirms that the large
    node-feature contribution was drowning out the geometry signal.

    Args: same signature as EdgeScoringNet for drop-in compatibility.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers: list = []
        dim = edge_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU()]
            dim = hidden_dim
        layers.append(nn.Linear(dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return per-edge logits [E, 2]; x is unused."""
        return self.net(y)


class LearnableCutoffScoringNet(nn.Module):
    """Learnable soft distance cutoff inspired by SchNet.

    keep_logit(r) = (r_cut - r) / tau
    Positive for bonds shorter than r_cut, negative for longer ones.
    r_cut and tau are scalar learnable parameters.

    Requires raw bond lengths stored in g.edata['bond_lengths'] (Angstrom).
    Args: same signature as EdgeScoringNet for drop-in compatibility.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_layers: int, init_r_cut: float = 3.5):
        super().__init__()
        self.r_cut = nn.Parameter(torch.tensor(float(init_r_cut)))  # learnable cutoff (Å)
        self.tau = nn.Parameter(torch.tensor(0.5))      # initial softness

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return per-edge logits [E, 2]."""
        r = g.edata['bond_lengths']   # [E]  (Angstrom)
        logit = (self.r_cut.abs() - r) / self.tau.abs().clamp(min=0.05)
        return torch.stack([logit, -logit], dim=-1)   # col-0 = keep, col-1 = drop


class MultiplicativeGateScoringNet(nn.Module):
    """Multiplicative distance-gate scorer inspired by DimeNet.

    gate(e_ij)     = sigmoid(dist_mlp(e_ij))     — geometry-driven [0,1] gate
    node_score(r)  = node_mlp([h_src || h_dst])   — chemistry-driven scalar
    logit(r)       = gate * node_score             — geometry modulates chemistry

    Short bonds produce large gate values, so the network is explicitly
    biased to attend to chemically significant (short) bonds.
    Args: same signature as EdgeScoringNet for drop-in compatibility.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.dist_gate = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return per-edge logits [E, 2]."""
        src, dst = g.edges()
        gate = self.dist_gate(y)                                           # [E, 1]
        node_score = self.node_mlp(torch.cat([x[src], x[dst]], dim=-1))   # [E, 1]
        logit = (gate * node_score).squeeze(-1)                            # [E]
        return torch.stack([logit, -logit], dim=-1)                        # [E, 2]


