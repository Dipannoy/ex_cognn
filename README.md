# CoGNN: Cooperative Graph Neural Networks for Crystal Property Prediction

Learned edge pruning for crystal graphs. A small **action network** decides, at every message-passing layer, which bonds each atom should listen to and broadcast on, so the property model ignores chemically irrelevant edges.

**Headline result:** keeps only **62–68% of crystal-graph edges** while cutting band gap error by **up to 2.8%** and matching or improving formation-energy accuracy. The pruned graphs also keep node representations more distinct at depth (less over-smoothing).

## Method

Standard message passing lets every atom exchange messages with all of its neighbours. CoGNN adds a cooperative layer on top of an ALIGNN-style backbone (`CoALIGNN`):

1. An **action network** scores each edge from the current atom and bond features.
2. Each atom chooses to *listen*, *broadcast*, *both*, or *isolate* on each edge; edges whose score falls below a threshold (`--edge_threshold`) are dropped for that layer.
3. The backbone runs message passing on the pruned graph, and the action network is trained end to end with the property loss.

Action-network variants in `models/alignn.py`: `ActionNet` (edge-gated), `ActionNet_mean_gnn_node`, `ActionNet_mean_gnn_node_edge`, `ActionNet_mean_gnn_weighted`.

Diagnostics:
- **MADGap** to measure over-smoothing
- **Jacobian sensitivity** to measure over-squashing
- Per-edge explanation tools in `graph_expln/`

## Results

| Task | Edges kept | Effect vs. full-graph baseline |
|---|---|---|
| Band gap | 62–68% | up to 2.8% lower MAE |
| Formation energy | 62–68% | similar or lower MAE |

Datasets: JARVIS-DFT, Materials Project, halide perovskites, and a polymer band-gap set.

## Repository layout

```
models/              CoALIGNN backbone, action networks, layers
main.py              train / evaluate one configuration
main_nfolds.py       k-fold cross-validation
runner.py            experiment runner
data.py, graphs.py   crystal-graph construction (JARVIS-tools + DGL)
graph_expln/         edge-level explanation scripts
datasets/            small preprocessed example datasets
```

## Installation

```bash
conda create -n cognn python=3.10 -y
conda activate cognn
pip install -r requirements.txt
```

Install the DGL build that matches your CUDA version from https://www.dgl.ai/pages/start.html.

## Quick start

```bash
python main.py \
  --root_dir ./datasets/mp/ef \
  --dataset_name mp_lg_unit_cell_ef.pickle \
  --target Ef \
  --cutoff 8 --max_neighbors 12 \
  --edge_threshold 0.1 \
  --batch_size 64 --epochs 300 \
  --output_dir ./runs/mp_ef
```

Use `main_nfolds.py` with the same arguments for cross-validation.

## Acknowledgements

The backbone builds on [ALIGNN](https://github.com/usnistgov/alignn) (NIST). The cooperative listen/broadcast idea follows [Cooperative Graph Neural Networks](https://arxiv.org/abs/2310.01267) (Finkelshtein et al., ICML 2024), adapted here to crystal graphs.

## Author

Dipannoy Das Gupta, University of South Carolina · [Google Scholar](https://scholar.google.com/citations?user=RUOyC1QAAAAJ&hl=en)
