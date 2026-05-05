from typing import Optional

import numpy as np
import torch
from torch import nn
import dgl

def batch_node_subgraph(bg: dgl.DGLGraph, mask: torch.Tensor) -> dgl.DGLGraph:      
    with bg.local_scope():        
        if mask.dtype != torch.bool:
            mask = mask.bool()
        bg.ndata['mask'] = mask
        sg = [dgl.node_subgraph(g, g.ndata.pop("mask")) 
              for g in dgl.unbatch(bg)]
        sg = dgl.batch(sg)
    return sg

def batch_edge_subgraph(bg: dgl.DGLGraph, mask: torch.Tensor) -> dgl.DGLGraph:
    with bg.local_scope():
        if mask.dtype != torch.bool:
            mask = mask.bool()
        bg.edata['mask'] = mask
        sg = [dgl.edge_subgraph(g, g.edata.pop("mask")) 
              for g in dgl.unbatch(bg)]
        sg = dgl.batch(sg)
    return sg


class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )
