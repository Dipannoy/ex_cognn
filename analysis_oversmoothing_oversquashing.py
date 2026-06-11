"""analysis_oversmoothing_oversquashing.py

Post-training analysis script.  Given a directory tree of trained models
(from the layer-sweep experiment), this script:

  1. Loads each trained checkpoint.
  2. Runs the test loader through AnalysisRunner.
  3. Computes MAD / MADGap  (oversmoothing) at each layer depth.
  4. Computes Jacobian sensitivity vs distance (oversquashing).
  5. Saves all results to a JSON file per experiment and a combined CSV.

Usage
-----
    python analysis_oversmoothing_oversquashing.py \\
        --results_root   outputs/layer_sweep \\
        --dataset        mp_eg \\
        --property       Eg \\
        --output_json    outputs/analysis/oversmoothing_oversquashing_mp_eg.json

The script auto-discovers experiment sub-directories that match the naming
pattern produced by run_layer_sweep_mp.sh:

    {dataset}_baseline_gc10_mn24_gn{LAYERS}_0607/
    {dataset}_lcsn_gc10_mn24_gn{LAYERS}_th{TH}_sw{SW}_0607/
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from data import get_loaders
from runner_analysis import AnalysisRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_best_checkpoint(runner: AnalysisRunner, output_dir: str, fold: int):
    """Load best-val checkpoint for a given fold into runner.model."""
    pattern = f"epoch=best={fold}.ckpt"
    ckpt_path = os.path.join(output_dir, pattern)
    if not os.path.exists(ckpt_path):
        # fall back: latest epoch checkpoint
        import glob
        candidates = sorted(glob.glob(os.path.join(output_dir, f"epoch=*={fold}.ckpt")))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found in {output_dir} for fold {fold}")
        ckpt_path = candidates[-1]

    ckpt = torch.load(ckpt_path, map_location=runner.device)
    runner.model.load_state_dict(ckpt["model_state_dict"])
    runner.model.eval()
    print(f"  Loaded {os.path.basename(ckpt_path)}")


def discover_experiments(results_root: str, dataset: str) -> list[dict]:
    """Scan results_root for layer-sweep output directories.

    Expected name patterns (from run_layer_sweep_mp.sh):
        {dataset}_baseline_gc10_mn24_gn{N}_0607
        {dataset}_lcsn_gc10_mn24_gn{N}_th{TH}_sw{SW}_0607
    """
    experiments = []
    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"results_root not found: {results_root}")

    for name in sorted(os.listdir(results_root)):
        if not name.startswith(dataset):
            continue
        path = os.path.join(results_root, name)
        if not os.path.isdir(path):
            continue
        # parse n_layers from directory name: gn{N}
        parts = name.split("_")
        gn_layers = None
        for p in parts:
            if p.startswith("gn") and p[2:].isdigit():
                gn_layers = int(p[2:])
        if gn_layers is None:
            continue
        model_type = "baseline" if "baseline" in name else "lcsn"
        experiments.append({
            "name":      name,
            "path":      path,
            "gn_layers": gn_layers,
            "model":     model_type,
        })
    return experiments


# def build_config_from_dir(exp_path: str) -> object:
#     """Load the OmegaConf config saved in the experiment directory."""
#     cfg_candidates = ["config.yaml", "config.json", "config_eg_coalignn.json",
#                       "config_ef_coalignn.json"]
#     for c in cfg_candidates:
#         p = os.path.join(exp_path, c)
#         if os.path.exists(p):
#             return OmegaConf.load(p)
#     # fall back: look for any .json/.yaml
#     import glob
#     for ext in ("*.yaml", "*.json"):
#         found = glob.glob(os.path.join(exp_path, ext))
#         if found:
#             return OmegaConf.load(found[0])
#     raise FileNotFoundError(f"No config file found in {exp_path}")


def build_config_from_dir(exp_path: str, args) -> object:
    """Load the base JSON config and patch all model parameters from the
    experiment directory name — mirrors what main.py does at training time.

    Layer-sweep naming convention (from run_layer_sweep_mp.sh):
        mp_{prop}_baseline_gc{C}_mn{N}_gn{L}_{TAG}
        mp_{prop}_lcsn_gc{C}_mn{N}_gn{L}_th{TH_TAG}_sw{SW_TAG}_{TAG}
    where TH_TAG / SW_TAG replace '.' with '_'.
    """
    import re as _re

    name = os.path.basename(exp_path)

    # ── Choose base config file ─────────────────────────────────────────────
    if "_ef_" in name or name.endswith("_ef"):
        cfg_path = os.path.join(args.datasets_root, "ef", "config_ef_coalignn.json")
    else:
        cfg_path = os.path.join(args.datasets_root, "config_eg_coalignn.json")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"Config not found: {cfg_path}\n"
            "Set --datasets_root to the datasets/mp parent directory."
        )
    cfg = OmegaConf.load(cfg_path)
    cfg.output_dir = exp_path   # checkpoints live in the output dir

    # ── Parse numeric tokens from directory name ────────────────────────────
    parts = name.split("_")

    def _find_int(prefix):
        for p in parts:
            if p.startswith(prefix) and p[len(prefix):].isdigit():
                return int(p[len(prefix):])
        return None

    gcn_layers    = _find_int("gn") or 4
    cutoff        = _find_int("gc") or 10
    max_neighbors = _find_int("mn") or 24

    model_type = "baseline" if "baseline" in name else "lcsn"

    # TH_TAG and SW_TAG use '_' in place of '.' — reconstruct with regex
    edge_threshold  = 0.0
    sparsity_weight = 0.0
    m_th = _re.search(r'_th([\d_]+)_sw', name)
    m_sw = _re.search(r'_sw([\d_]+)(?:_\d{4}|_[a-z]|$)', name)
    if m_th:
        edge_threshold = float(m_th.group(1).replace("_", "."))
    if m_sw:
        sparsity_weight = float(m_sw.group(1).replace("_", "."))

    # ── Patch config — mirrors main.py's main() function ───────────────────
    # Fixed for all layer-sweep jobs (from run_layer_sweep_mp.sh):
    #   --aln_layer 0  --anlg_layer 0  --in_act_graph_layer 1
    #   --action_net LearnableCutoffScoringNet
    cfg.model.alignn_layers                        = 0
    cfg.model.gcn_layers                           = gcn_layers
    cfg.model.edge_threshold                       = edge_threshold
    cfg.model.check_threshold                      = (model_type == "lcsn")
    cfg.model.action_args.num_act_graph_layers     = 1 if model_type == "lcsn" else 0
    cfg.model.action_args.num_act_linegraph_layers = 0
    cfg.model.action_args.num_in_act_graph_layers  = 1
    cfg.model.action_args.action_net               = "LearnableCutoffScoringNet"
    cfg.model.random_weight                        = 0
    cfg.model.logit_1                              = 0.5
    cfg.model.logit_2                              = 0.5
    cfg.model.init_r_cut                           = 3.5
    cfg.sparsity_weight                            = sparsity_weight
    cfg.cutoff                                     = cutoff
    cfg.max_neighbors                              = max_neighbors
    cfg.neighbor_strategy                          = "k-nearest"

    print(f"  [build_config] {name}")
    print(f"    model={model_type}  gcn={gcn_layers}  gc={cutoff}  mn={max_neighbors}"
          f"  th={edge_threshold}  sw={sparsity_weight}")

    return cfg

# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def _agg_mad(fold_mad_list):
    keys = ["mad_all", "mad_local", "mad_remote", "mad_gap"]
    return {k: float(np.mean([d[k] for d in fold_mad_list if not np.isnan(d[k])]))
            for k in keys}


def _agg_sens(fold_sens_list):
    means = np.array([s["mean"] for s in fold_sens_list])  # [folds, D+1]
    return {
        "mean":   means.mean(axis=0).tolist(),
        "std":    means.std(axis=0).tolist(),
        "counts": np.array([s["counts"] for s in fold_sens_list]).sum(axis=0).tolist(),
    }


def _analyse_one(exp: dict, args) -> dict:
    """Run MAD + Jacobian analysis for a single experiment. Returns one record."""
    print(f"\n{'='*60}")
    print(f"Analysing: {exp['name']}")
    print(f"{'='*60}")

    cfg    = build_config_from_dir(exp["path"], args)
    runner = AnalysisRunner(config=cfg, verbose=False)

    fold_mae, fold_mad, fold_sens = [], [], []

    for fold in range(args.n_folds):
        try:
            load_best_checkpoint(runner, exp["path"], fold)
        except FileNotFoundError as exc:
            print(f"  [fold {fold}] SKIP: {exc}")
            continue

        loaders = get_loaders(
            config         = cfg,
            root_dir       = args.root_dir,
            sub_dir        = None,
            dataset_name   = args.dataset_name,
            target_file    = args.target_file,
            fold           = fold,
            trim_graph_dir = None,
        )
        test_loader = loaders[2]   # train, val, test

        # MAE from saved error.csv
        error_csv = os.path.join(exp["path"], "error.csv")
        if os.path.exists(error_csv):
            df_err   = pd.read_csv(error_csv)
            test_row = df_err[df_err["sample"] == "test"]
            if not test_row.empty:
                fold_mae.append(float(test_row["mean"].values[0]))

        # MAD (oversmoothing)
        t0 = time.time()
        mad_dict = runner.compute_mad(test_loader,
                                      max_structures=args.max_mad_structures)
        print(f"  [fold {fold}] MAD={mad_dict['mad_all']:.4f}  "
              f"MADGap={mad_dict['mad_gap']:.4f}  ({time.time()-t0:.1f} s)")
        fold_mad.append(mad_dict)

        # Jacobian sensitivity (oversquashing)
        t0 = time.time()
        sens_dict = runner.compute_jacobian_sensitivity(
            test_loader,
            n_structures = args.n_jacobian_structures,
            max_distance = args.max_distance,
        )
        print(f"  [fold {fold}] sens@d=1:{sens_dict['mean'][1]:.4e}  "
              f"sens@d=2:{sens_dict['mean'][2]:.4e}  ({time.time()-t0:.1f} s)")
        fold_sens.append(sens_dict)

    rec = {
        "name":         exp["name"],
        "model":        exp["model"],
        "gn_layers":    exp["gn_layers"],
        "test_mae":     float(np.mean(fold_mae))   if fold_mae  else None,
        "test_mae_std": float(np.std(fold_mae))    if fold_mae  else None,
        "mad":          _agg_mad(fold_mad)          if fold_mad  else {},
        "sensitivity":  _agg_sens(fold_sens)        if fold_sens else {},
    }
    print(f"  → test MAE = {rec['test_mae']}")
    return rec


def _save_results(results: list, output_json: str):
    """Write combined JSON + flat CSV."""
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_json}")

    csv_path = output_json.replace(".json", ".csv")
    rows = []
    for r in results:
        rows.append({
            "name":         r["name"],
            "model":        r["model"],
            "gn_layers":    r["gn_layers"],
            "test_mae":     r.get("test_mae"),
            "test_mae_std": r.get("test_mae_std"),
            "mad_all":      r["mad"].get("mad_all"),
            "mad_local":    r["mad"].get("mad_local"),
            "mad_remote":   r["mad"].get("mad_remote"),
            "mad_gap":      r["mad"].get("mad_gap"),
            "sens_d1":      r["sensitivity"].get("mean", [None]*2)[1] if r.get("sensitivity") else None,
            "sens_d2":      r["sensitivity"].get("mean", [None]*3)[2] if r.get("sensitivity") else None,
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")


def run_analysis(args):
    # ── Single-experiment (shard) mode ── called from each SLURM job ────────
    # Analyses only the given experiment directory and writes a private shard
    # file.  No interaction with other jobs → no race conditions.
    if args.exp_dir:
        exp_dir   = args.exp_dir.rstrip("/")
        exp_name  = os.path.basename(exp_dir)
        parts     = exp_name.split("_")
        gn_layers = next((int(p[2:]) for p in parts
                          if p.startswith("gn") and p[2:].isdigit()), None)
        model_type = "baseline" if "baseline" in exp_name else "lcsn"
        exp = {"name": exp_name, "path": exp_dir,
               "gn_layers": gn_layers, "model": model_type}

        rec = _analyse_one(exp, args)

        shard_dir  = os.path.dirname(args.output_json) or "."
        shard_base = os.path.splitext(os.path.basename(args.output_json))[0]
        shard_path = os.path.join(shard_dir, f"{shard_base}_{exp_name}.json")
        os.makedirs(shard_dir, exist_ok=True)
        with open(shard_path, "w") as f:
            json.dump([rec], f, indent=2)
        print(f"Shard saved: {shard_path}")
        return

    # ── Aggregation mode ── called once after all jobs finish ────────────────
    # Merges all per-experiment shard files into one combined JSON + CSV.
    # Missing shards are skipped with a warning (job not yet finished).
    if not args.results_root:
        print("ERROR: provide --exp_dir (per-job/shard mode) "
              "or --results_root (aggregate mode).")
        sys.exit(1)

    experiments = discover_experiments(args.results_root, args.dataset)
    if not experiments:
        print(f"No experiments found in {args.results_root} for '{args.dataset}'")
        sys.exit(1)

    print(f"Found {len(experiments)} experiment(s):")
    for e in experiments:
        print(f"  [{e['model']:8s} gn={e['gn_layers']}]  {e['name']}")

    shard_dir  = os.path.dirname(args.output_json) or "."
    shard_base = os.path.splitext(os.path.basename(args.output_json))[0]

    results = []
    for exp in experiments:
        shard_path = os.path.join(shard_dir, f"{shard_base}_{exp['name']}.json")
        if os.path.exists(shard_path):
            with open(shard_path) as f:
                results.extend(json.load(f))
            print(f"  Loaded shard: {os.path.basename(shard_path)}")
        else:
            print(f"  WARNING: no shard for {exp['name']} — skipping "
                  f"(run its SLURM job first).")

    if not results:
        print("No shards found. Run all SLURM jobs before aggregating.")
        sys.exit(1)

    results.sort(key=lambda r: (r["model"], r["gn_layers"]))
    _save_results(results, args.output_json)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Oversmoothing & oversquashing analysis for CoALIGNN layer sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Mode selection (mutually exclusive) ─────────────────────────────────
    p.add_argument("--exp_dir",        default=None,
                   help="[Shard mode] Analyse a single experiment directory "
                        "and write a per-experiment shard JSON. "
                        "Used by each SLURM job.")
    p.add_argument("--results_root",   default=None,
                   help="[Aggregate mode] Root directory containing layer-sweep "
                        "output folders. Merges existing shards into one JSON.")
    p.add_argument("--dataset",        default="mp_eg",
                   help="Dataset prefix used in directory names (e.g. mp_eg, mp_ef). "
                        "Only used in aggregate mode.")
    # ── Data / checkpoint paths ─────────────────────────────────────────────
    p.add_argument("--datasets_root",  default="datasets/mp",
                   help="Path to the MP datasets directory "
                        "(contains config_eg_coalignn.json and ef/).")
    p.add_argument("--root_dir",       default="datasets/mp",
                   help="Root dir for the dataset pickle and CSV.")
    p.add_argument("--dataset_name",   default="mp_lg_unit_cell_gap.pickle")
    p.add_argument("--target_file",    default="mp_gap_lg_unit_cell_id.csv")
    # ── Analysis knobs ──────────────────────────────────────────────────────
    p.add_argument("--n_folds",        default=5, type=int)
    p.add_argument("--max_mad_structures",    default=None, type=int,
                   help="Limit MAD computation to N structures per fold (None=all).")
    p.add_argument("--n_jacobian_structures", default=60, type=int,
                   help="Number of test structures for Jacobian analysis per fold.")
    p.add_argument("--max_distance",          default=6, type=int,
                   help="Maximum graph distance to consider for Jacobian binning.")
    # ── Output ──────────────────────────────────────────────────────────────
    p.add_argument("--output_json",    default="outputs/analysis/oversmoothing_oversquashing_mp_eg.json",
                   help="Combined output JSON (aggregate mode) or shard naming base (shard mode).")
    return p.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
