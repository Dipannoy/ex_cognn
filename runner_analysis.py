"""runner_analysis.py

Analysis runner for oversmoothing and oversquashing diagnostics.

Two metrics are computed on a *trained* model without any retraining:

Oversmoothing — Mean Average Distance (MAD) and MADGap
    MAD measures how distinct node representations remain after k GCN layers.
    A well-trained model should have high MADGap (remote pairs are less similar
    than local pairs). As oversmoothing sets in, MADGap → 0.

        MAD_all   = mean(1 - cos_sim(h_i, h_j))  over all node pairs (i,j)
        MAD_local = mean(...)                      restricted to directly connected pairs
        MAD_remote= mean(...)                      restricted to non-connected pairs
        MADGap    = MAD_remote - MAD_local

Oversquashing — Jacobian sensitivity vs graph distance
    Sensitivity of the graph-level prediction ŷ to each node's input features,
    grouped by the shortest-path distance d(i, node_of_interest):

        S_k = mean_{all i, j: dist(i,j)=k}  ||∂ŷ / ∂x_j||_2

    Exponential decay of S_k with k is the signature of oversquashing.
    CoALIGNN (pruned graph) should maintain higher sensitivity at distance ≥ 2
    compared to the dense baseline.
"""

from __future__ import annotations

import numpy as np
import torch
import dgl

from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from runner import Runner


# ---------------------------------------------------------------------------
# MAD helpers
# ---------------------------------------------------------------------------

def _cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Return N×N cosine similarity matrix for rows of X."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    X_norm = X / norms
    return X_norm @ X_norm.T          # [N, N]


def compute_mad_for_structure(embeddings: np.ndarray,
                               src: np.ndarray,
                               dst: np.ndarray) -> dict:
    """Compute MAD, MAD_local, MAD_remote, MADGap for one crystal.

    Parameters
    ----------
    embeddings : [N, H] node embeddings
    src, dst   : edge endpoints (undirected: both (u,v) and (v,u) present)
    """
    N = embeddings.shape[0]
    if N < 2:
        return None

    sim = _cosine_similarity_matrix(embeddings)   # [N, N]
    dist = 1.0 - sim                               # dissimilarity in [0,2]

    # adjacency mask (local = directly connected)
    adj = np.zeros((N, N), dtype=bool)
    adj[src, dst] = True

    # exclude self-pairs
    self_mask = np.eye(N, dtype=bool)
    all_mask   = ~self_mask

    local_mask  = adj & ~self_mask
    remote_mask = ~adj & ~self_mask

    mad_all    = dist[all_mask].mean()    if all_mask.any()    else np.nan
    mad_local  = dist[local_mask].mean()  if local_mask.any()  else np.nan
    mad_remote = dist[remote_mask].mean() if remote_mask.any() else np.nan
    mad_gap    = mad_remote - mad_local

    return {"mad_all": mad_all, "mad_local": mad_local,
            "mad_remote": mad_remote, "mad_gap": mad_gap, "n_nodes": N}


def aggregate_mad(mad_list: list[dict]) -> dict:
    """Weighted mean (by n_nodes) over per-structure MAD dicts."""
    keys = ["mad_all", "mad_local", "mad_remote", "mad_gap"]
    totals = defaultdict(float)
    weights = defaultdict(float)
    for d in mad_list:
        if d is None:
            continue
        w = d["n_nodes"]
        for k in keys:
            if not np.isnan(d[k]):
                totals[k]  += d[k] * w
                weights[k] += w
    return {k: (totals[k] / weights[k] if weights[k] > 0 else np.nan)
            for k in keys}


# ---------------------------------------------------------------------------
# Rank-based oversmoothing metrics
#
# Zhang, Deidda, Higham & Tudisco, "Are We Measuring Oversmoothing in Graph
# Neural Networks Correctly?" (ICLR 2026) argue that Dirichlet/MAD-like
# energy metrics only signal oversmoothing once they converge exactly to
# zero, which in practice requires very deep, weight-bounded networks —
# realistic shallow-to-moderate-depth models can suffer clear performance
# degradation with no visible change in MAD. They propose measuring the
# (continuous relaxation of the) rank of the feature matrix instead, since
# it is scale-invariant and detects convergence toward *any* low-rank
# subspace, not just exact alignment with a fixed dominant eigenvector.
#
#   NumRank(X) = ||X||_F^2 / ||X||_2^2     (numerical rank)
#   Erank(X)   = exp(-sum_k p_k log p_k),  p_k = sigma_k / sum(sigma)
#                                            (Shannon / effective rank)
#
# Both equal 1 exactly at rank-1 collapse and grow toward min(N, H) as the
# embedding matrix retains more independent directions of variation, so a
# decreasing trend with depth is the oversmoothing signal (analogous to a
# shrinking MADGap, but without requiring exact convergence to detect it).
# ---------------------------------------------------------------------------

def compute_rank_metrics_for_structure(embeddings: np.ndarray) -> dict:
    """NumRank and Erank of one crystal's final-layer node embedding matrix."""
    N = embeddings.shape[0]
    if N < 2:
        return None

    sigma = np.linalg.svd(embeddings, compute_uv=False)
    sigma = sigma[sigma > 1e-12]
    if len(sigma) == 0:
        return None

    num_rank = float(np.sum(sigma ** 2) / sigma[0] ** 2)

    p = sigma / sigma.sum()
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))

    return {"numerical_rank": num_rank, "effective_rank": eff_rank, "n_nodes": N}


def aggregate_rank_metrics(rank_list: list[dict]) -> dict:
    """Weighted mean (by n_nodes) over per-structure rank-metric dicts."""
    keys = ["numerical_rank", "effective_rank"]
    totals = defaultdict(float)
    weights = defaultdict(float)
    for d in rank_list:
        if d is None:
            continue
        w = d["n_nodes"]
        for k in keys:
            totals[k]  += d[k] * w
            weights[k] += w
    return {k: (totals[k] / weights[k] if weights[k] > 0 else np.nan)
            for k in keys}


# ---------------------------------------------------------------------------
# Jacobian sensitivity helpers
# ---------------------------------------------------------------------------

def _bfs_distances(src: np.ndarray, dst: np.ndarray, n_nodes: int) -> np.ndarray:
    """Return [N, N] integer shortest-path distance matrix (0 on diagonal)."""
    data = np.ones(len(src), dtype=np.float32)
    # build undirected CSR
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    dat = np.ones(len(row), dtype=np.float32)
    A = csr_matrix((dat, (row, col)), shape=(n_nodes, n_nodes))
    D = shortest_path(A, directed=False, unweighted=True)
    return D   # [N, N], np.inf for unreachable


class AnalysisRunner(Runner):
    """Runner extended with MAD and Jacobian sensitivity analysis methods."""

    # ------------------------------------------------------------------ MAD --

    def compute_mad(self, loader, max_structures: int = None) -> dict:
        """Compute MAD / MADGap across test structures.

        Parameters
        ----------
        loader        : DataLoader (test split)
        max_structures: stop after this many crystals (None = all)

        Returns
        -------
        dict with keys: mad_all, mad_local, mad_remote, mad_gap
        """
        embeddings_by_jid = self.extract_node_embeddings(loader)

        mad_list = []
        for jid, info in embeddings_by_jid.items():
            m = compute_mad_for_structure(
                info["embeddings"], info["src"], info["dst"]
            )
            mad_list.append(m)
            if max_structures and len(mad_list) >= max_structures:
                break

        return aggregate_mad(mad_list)

    # ------------------------------------------------------------- Rank --

    def compute_rank_metrics(self, loader, max_structures: int = None) -> dict:
        """Compute NumRank / Erank across test structures (Zhang et al. 2026).

        Parameters
        ----------
        loader        : DataLoader (test split)
        max_structures: stop after this many crystals (None = all)

        Returns
        -------
        dict with keys: numerical_rank, effective_rank
        """
        embeddings_by_jid = self.extract_node_embeddings(loader)

        rank_list = []
        for jid, info in embeddings_by_jid.items():
            r = compute_rank_metrics_for_structure(info["embeddings"])
            rank_list.append(r)
            if max_structures and len(rank_list) >= max_structures:
                break

        return aggregate_rank_metrics(rank_list)

    # -------------------------------------------------------- Jacobian sens. --

    def compute_jacobian_sensitivity(self,
                                     loader,
                                     n_structures: int = 80,
                                     max_distance: int = 6) -> dict:
        """Compute mean ||∂ŷ/∂x_j|| vs shortest-path distance.

        For each structure we pick the node with the highest embedding norm
        as the "reference" node i, compute ∂ŷ/∂x_j for all j, and bin by
        d(i,j).  Results are averaged across structures.

        Parameters
        ----------
        loader        : DataLoader (test split, batch_size should be 1 for
                        exact per-structure Jacobians; larger batches are also
                        supported but will mix structures)
        n_structures  : how many structures to analyse
        max_distance  : distances beyond this are ignored

        Returns
        -------
        dict  {"mean": np.ndarray [max_distance+1],
               "std":  np.ndarray [max_distance+1],
               "counts": np.ndarray [max_distance+1]}
        """
        self.model.eval()

        sens_by_dist: dict[int, list[float]] = defaultdict(list)
        n_done = 0

        for batch in loader:
            if n_done >= n_structures:
                break

            g_batch, lg_batch, y_batch = batch
            g_batch  = g_batch.to(self.device)
            lg_batch = lg_batch.to(self.device)

            # un-batch: process one structure at a time
            graphs  = dgl.unbatch(g_batch)
            lgs     = dgl.unbatch(lg_batch)

            for g_single, lg_single in zip(graphs, lgs):
                if n_done >= n_structures:
                    break

                n_nodes = g_single.num_nodes()
                if n_nodes < 3:
                    continue

                src_np = g_single.edges()[0].cpu().numpy()
                dst_np = g_single.edges()[1].cpu().numpy()
                D = _bfs_distances(src_np, dst_np, n_nodes)   # [N, N]

                # Enable grad on node features
                x_orig = g_single.ndata["atom_features"].clone().float()
                x_in   = x_orig.to(self.device).requires_grad_(True)
                g_single.ndata["atom_features"] = x_in

                g_single  = g_single.to(self.device)
                lg_single = lg_single.to(self.device)

                with torch.enable_grad():
                    out, _, _, _, _ = self.model(
                        (g_single, lg_single),
                        keep_bond_forcing=False,
                        context=torch.enable_grad(),
                        is_hard=False,
                        is_eval=True,
                    )
                    # scalar prediction → sum is identity for single output
                    scalar_out = out.sum()
                    scalar_out.backward()

                grad = x_in.grad.detach().cpu()          # [N, F]
                sens = grad.norm(dim=1).numpy()          # [N]  per-node sensitivity

                # reference node: pick node with max output sensitivity
                # (alternatively the most "central" node — both work)
                ref = int(np.argmax(sens))

                for j in range(n_nodes):
                    d = int(D[ref, j])
                    if d == 0 or d > max_distance or np.isinf(D[ref, j]):
                        continue
                    sens_by_dist[d].append(float(sens[j]))

                # restore atom_features for next pass
                g_single.ndata["atom_features"] = x_orig.to(self.device)
                n_done += 1

        mean_arr   = np.zeros(max_distance + 1)
        std_arr    = np.zeros(max_distance + 1)
        count_arr  = np.zeros(max_distance + 1, dtype=int)
        for d in range(1, max_distance + 1):
            vals = sens_by_dist[d]
            if vals:
                mean_arr[d]  = float(np.mean(vals))
                std_arr[d]   = float(np.std(vals))
                count_arr[d] = len(vals)

        return {"mean": mean_arr, "std": std_arr, "counts": count_arr}
