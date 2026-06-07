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


def build_config_from_dir(exp_path: str) -> object:
    """Load the OmegaConf config saved in the experiment directory."""
    cfg_candidates = ["config.yaml", "config.json", "config_eg_coalignn.json",
                      "config_ef_coalignn.json"]
    for c in cfg_candidates:
        p = os.path.join(exp_path, c)
        if os.path.exists(p):
            return OmegaConf.load(p)
    # fall back: look for any .json/.yaml
    import glob
    for ext in ("*.yaml", "*.json"):
        found = glob.glob(os.path.join(exp_path, ext))
        if found:
            return OmegaConf.load(found[0])
    raise FileNotFoundError(f"No config file found in {exp_path}")


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def run_analysis(args):
    results = []

    experiments = discover_experiments(args.results_root, args.dataset)
    if not experiments:
        print(f"No experiments found in {args.results_root} for dataset '{args.dataset}'")
        sys.exit(1)

    print(f"Found {len(experiments)} experiment(s):")
    for e in experiments:
        print(f"  [{e['model']:8s} gn={e['gn_layers']}]  {e['name']}")

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Analysing: {exp['name']}")
        print(f"{'='*60}")

        cfg = build_config_from_dir(exp["path"])
        cfg.output_dir = exp["path"]

        runner = AnalysisRunner(config=cfg, verbose=False)

        # MAE is averaged over folds; we also average MAD / sensitivity over folds
        fold_mae, fold_mad, fold_sens = [], [], []
        n_folds = args.n_folds

        for fold in range(n_folds):
            try:
                load_best_checkpoint(runner, exp["path"], fold)
            except FileNotFoundError as e:
                print(f"  [fold {fold}] SKIP: {e}")
                continue

            loaders = get_loaders(
                config       = cfg,
                root_dir     = args.root_dir,
                sub_dir      = None,
                dataset_name = args.dataset_name,
                target_file  = args.target_file,
                fold         = fold,
                trim_graph_dir = None,
            )
            test_loader = loaders[2]   # train, val, test, _

            # -- MAE (from saved error.csv if available, else recompute) ----
            error_csv = os.path.join(exp["path"], "error.csv")
            if os.path.exists(error_csv):
                df_err = pd.read_csv(error_csv)
                test_row = df_err[df_err["sample"] == "test"]
                if not test_row.empty:
                    fold_mae.append(float(test_row["mean"].values[0]))

            # -- MAD (oversmoothing) ----------------------------------------
            t0 = time.time()
            mad_dict = runner.compute_mad(test_loader,
                                          max_structures=args.max_mad_structures)
            print(f"  [fold {fold}] MAD={mad_dict['mad_all']:.4f}  "
                  f"MADGap={mad_dict['mad_gap']:.4f}  "
                  f"({time.time()-t0:.1f} s)")
            fold_mad.append(mad_dict)

            # -- Jacobian sensitivity (oversquashing) -----------------------
            t0 = time.time()
            sens_dict = runner.compute_jacobian_sensitivity(
                test_loader,
                n_structures = args.n_jacobian_structures,
                max_distance = args.max_distance,
            )
            print(f"  [fold {fold}] sens@d=1:{sens_dict['mean'][1]:.4e}  "
                  f"sens@d=2:{sens_dict['mean'][2]:.4e}  "
                  f"({time.time()-t0:.1f} s)")
            fold_sens.append(sens_dict)

        # -- Aggregate over folds ------------------------------------------
        def _agg_mad(fold_mad_list):
            keys = ["mad_all", "mad_local", "mad_remote", "mad_gap"]
            return {k: float(np.mean([d[k] for d in fold_mad_list
                                      if not np.isnan(d[k])]))
                    for k in keys}

        def _agg_sens(fold_sens_list, max_d):
            means = np.array([s["mean"] for s in fold_sens_list])  # [folds, D+1]
            return {
                "mean":   means.mean(axis=0).tolist(),
                "std":    means.std(axis=0).tolist(),
                "counts": np.array([s["counts"] for s in fold_sens_list])
                            .sum(axis=0).tolist(),
            }

        rec = {
            "name":      exp["name"],
            "model":     exp["model"],
            "gn_layers": exp["gn_layers"],
            "test_mae":  float(np.mean(fold_mae)) if fold_mae else None,
            "test_mae_std": float(np.std(fold_mae)) if fold_mae else None,
            "mad":       _agg_mad(fold_mad)  if fold_mad  else {},
            "sensitivity": _agg_sens(fold_sens, args.max_distance) if fold_sens else {},
        }
        results.append(rec)
        print(f"  → test MAE = {rec['test_mae']}")

    # -- Save JSON -----------------------------------------------------------
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_json}")

    # -- Save flat CSV -------------------------------------------------------
    csv_path = args.output_json.replace(".json", ".csv")
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Oversmoothing & oversquashing analysis for CoALIGNN layer sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_root",   required=True,
                   help="Root directory containing layer-sweep output folders.")
    p.add_argument("--dataset",        default="mp_eg",
                   help="Dataset prefix used in directory names (e.g. mp_eg, mp_ef).")
    p.add_argument("--root_dir",       default="datasets/mp",
                   help="Root dir for the dataset pickle and CSV.")
    p.add_argument("--dataset_name",   default="mp_lg_unit_cell_gap.pickle")
    p.add_argument("--target_file",    default="mp_gap_lg_unit_cell_id.csv")
    p.add_argument("--n_folds",        default=5, type=int)
    p.add_argument("--output_json",    default="outputs/analysis/oversmoothing_oversquashing_mp_eg.json")
    p.add_argument("--max_mad_structures",    default=None, type=int,
                   help="Limit MAD computation to N structures per fold (None=all).")
    p.add_argument("--n_jacobian_structures", default=60, type=int,
                   help="Number of test structures for Jacobian analysis per fold.")
    p.add_argument("--max_distance",          default=6, type=int,
                   help="Maximum graph distance to consider for Jacobian binning.")
    return p.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
