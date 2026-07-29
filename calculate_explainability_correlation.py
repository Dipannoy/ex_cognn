"""
Calculate correlation metrics between bond length and edge weight
for all explainability experiments, aggregated across all CV folds.

Metrics reported:
  - Pearson r          (linear correlation)
  - Spearman rho       (monotonic / rank correlation)
  - R²                 (coefficient of determination from OLS linear fit)
  - Slope & intercept  (OLS linear regression)

Usage:
    python3 calculate_explainability_correlation.py
"""
from __future__ import annotations

import os
import json
import warnings
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import r2_score, roc_auc_score

# ---------------------------------------------------------------------------
# Mirror the exact experiment definitions from make_explainability_figure_all_fold.py
# ---------------------------------------------------------------------------
BASE = os.path.join(os.path.dirname(__file__), "downloaded_results/outputs_latest_data")

EXPERIMENTS = [
    {
        "label":     "JARVIS E_g EOSN",
        "threshold": 0.1,
        "dir":       os.path.join(BASE, "mp_eg_edge_only_scoring_net_th0_1_sw0_001"),
        "json_key":  "test_mp_large_eg",
    },
    {
        "label":     "JARVIS E_g ESN",
        "threshold": 0.1,
        "dir":       os.path.join(BASE, "mp_eg_edge_scoring_net_th0_1_sw0_001"),
        "json_key":  "test_mp_large_eg",
    },
    {
        "label":     "JARVIS E_g MGSN",
        "threshold": 0.1,
        "dir":       os.path.join(BASE, "mp_eg_multiplicative_gate_scoring_net_th0_1_sw0_001"),
        "json_key":  "test_mp_large_eg",
    },
    {
        "label":     "JARVIS E_g LCSN",
        "threshold": 0.2,
        "dir":       os.path.join(BASE, "mp_eg_learnable_cutoff_scoring_net_th0_2_sw0_001"),
        "json_key":  "test_mp_large_eg",
    },
]

N_FOLDS = 5


# ---------------------------------------------------------------------------
# Load & aggregate all fold JSON files for one experiment
# ---------------------------------------------------------------------------
def load_all_folds(exp: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    key     = exp["json_key"]
    out_dir = exp["dir"]

    all_weights: list[float] = []
    all_lengths: list[float] = []
    folds_found: list[int]   = []

    for fold in range(N_FOLDS):
        candidates = [
            os.path.join(out_dir, f"test_{key}_{fold}_prob_dict.json"),
            os.path.join(out_dir, f"{key}_test_{fold}_prob_dict.json"),
            os.path.join(out_dir, f"{key}_{fold}_prob_dict.json"),
        ]
        json_path = next((p for p in candidates if os.path.exists(p)), None)

        if json_path is None:
            print(f"    [fold {fold}] WARNING: no JSON found — skipping.")
            continue

        with open(json_path) as f:
            prob_data = json.load(f)

        n_before = len(all_weights)
        for jid, data in prob_data.items():
            weights = data.get("prop_dist", [])
            lengths = data.get("bond_lengths", [])
            if len(weights) == len(lengths) and len(weights) > 0:
                all_weights.extend(weights)
                all_lengths.extend(lengths)

        n_added = len(all_weights) - n_before
        print(f"    [fold {fold}] Added {n_added:,} edges (running total: {len(all_weights):,})")
        folds_found.append(fold)

    return (
        np.array(all_weights, dtype=float),
        np.array(all_lengths, dtype=float),
        folds_found,
    )


# ---------------------------------------------------------------------------
# Compute correlation metrics between bond length (x) and edge weight (y)
# ---------------------------------------------------------------------------
def compute_correlations(lengths: np.ndarray, weights: np.ndarray) -> dict:
    """Return a dict of correlation metrics between lengths and weights."""
    # Remove any NaN / inf rows
    mask   = np.isfinite(lengths) & np.isfinite(weights)
    x, y   = lengths[mask], weights[mask]

    # Constant-input guard: correlation is undefined if either array is constant
    if np.std(x) == 0 or np.std(y) == 0:
        return {
            "n_edges":      int(mask.sum()),
            "pearson_r":    float("nan"),
            "pearson_p":    float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p":   float("nan"),
            "r2":           float("nan"),
            "slope":        float("nan"),
            "intercept":    float("nan"),
            "stderr":       float("nan"),
            "note":         "constant input — correlation undefined",
        }

    # Linear regression via scipy (gives slope, intercept, r, p-value, stderr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope, intercept, pearson_r, p_value, stderr = stats.linregress(x, y)
        spearman_rho, spearman_p = stats.spearmanr(x, y)

    y_pred = slope * x + intercept
    r2     = r2_score(y, y_pred)

    return {
        "n_edges":      int(mask.sum()),
        "pearson_r":    float(pearson_r),
        "pearson_p":    float(p_value),
        "spearman_rho": float(spearman_rho),
        "spearman_p":   float(spearman_p),
        "r2":           float(r2),
        "slope":        float(slope),
        "intercept":    float(intercept),
        "stderr":       float(stderr),
    }


def compute_kept_dropped_correlations(
    lengths: np.ndarray, weights: np.ndarray, threshold: float
) -> dict:
    """Compute correlations separately for kept and dropped edges."""
    kept_mask    = weights >= threshold
    dropped_mask = ~kept_mask

    results = {}
    for subset_name, mask in [("kept", kept_mask), ("dropped", dropped_mask)]:
        if mask.sum() < 10:
            results[subset_name] = None
            continue
        results[subset_name] = compute_correlations(lengths[mask], weights[mask])
    return results


# ---------------------------------------------------------------------------
# Distributional separability: kept vs dropped bond-length distributions
# ---------------------------------------------------------------------------
def compute_separation_metrics(
    lengths: np.ndarray, weights: np.ndarray, threshold: float,
    n_bins: int = 200,
) -> dict:
    """Quantify how separable the kept / dropped bond-length distributions are.

    Parameters
    ----------
    lengths   : bond lengths for all edges
    weights   : edge weights (prop_dist) for all edges
    threshold : keep/drop decision boundary
    n_bins    : histogram resolution for density-based metrics

    Returns
    -------
    dict with keys:
      n_kept, n_dropped,
      ovl            -- Overlap Coefficient   [0=no overlap, 1=identical]
      auroc          -- AUROC (bond len -> kept label)  [0.5=random, 1=perfect]
      jsd            -- Jensen-Shannon Divergence       [0=identical, 1=fully apart]
      bhattacharyya  -- Bhattacharyya Coefficient       [0=no overlap, 1=identical]
      cohens_d       -- Cohen's d effect size
      mean_kept, mean_dropped, std_kept, std_dropped
    """
    kept_mask    = np.isfinite(lengths) & np.isfinite(weights) & (weights >= threshold)
    dropped_mask = np.isfinite(lengths) & np.isfinite(weights) & (weights < threshold)

    x_kept    = lengths[kept_mask]
    x_dropped = lengths[dropped_mask]

    if len(x_kept) < 10 or len(x_dropped) < 10:
        return {"note": "insufficient data"}

    # ---- Shared bin grid across both populations ----
    x_min = min(x_kept.min(), x_dropped.min())
    x_max = max(x_kept.max(), x_dropped.max())
    bins  = np.linspace(x_min, x_max, n_bins + 1)

    h_kept,    _ = np.histogram(x_kept,    bins=bins, density=True)
    h_dropped, _ = np.histogram(x_dropped, bins=bins, density=True)
    bin_width     = bins[1] - bins[0]

    # Normalise to proper probability mass (sum to 1)
    p_kept    = h_kept    * bin_width
    p_dropped = h_dropped * bin_width
    # Guard against floating-point drift
    p_kept    = np.clip(p_kept,    0, None);  p_kept    /= p_kept.sum()
    p_dropped = np.clip(p_dropped, 0, None);  p_dropped /= p_dropped.sum()

    # Overlap Coefficient
    ovl = float(np.minimum(p_kept, p_dropped).sum())

    # Jensen-Shannon Divergence (scipy returns JSD distance = sqrt(JSD); square it)
    jsd_dist = jensenshannon(p_kept, p_dropped, base=2)   # base-2: range [0, 1]
    jsd      = float(jsd_dist ** 2)                        # divergence, not distance

    # Bhattacharyya Coefficient  BC = sum( sqrt(p * q) )
    bc = float(np.sum(np.sqrt(p_kept * p_dropped)))

    # AUROC: label kept=1, dropped=0; score = bond length (negated because shorter=kept)
    # We use -length so higher score => more likely kept (conventional AUROC direction)
    n_kept_s    = len(x_kept)
    n_dropped_s = len(x_dropped)
    # Subsample to max 500k to keep it fast
    MAX_AUROC = 500_000
    if n_kept_s + n_dropped_s > MAX_AUROC:
        rng  = np.random.default_rng(42)
        frac = MAX_AUROC / (n_kept_s + n_dropped_s)
        idx_k = rng.choice(n_kept_s,    size=int(n_kept_s    * frac), replace=False)
        idx_d = rng.choice(n_dropped_s, size=int(n_dropped_s * frac), replace=False)
        scores = np.concatenate([-x_kept[idx_k], -x_dropped[idx_d]])
        labels = np.concatenate([np.ones(len(idx_k)), np.zeros(len(idx_d))])
    else:
        scores = np.concatenate([-x_kept, -x_dropped])
        labels = np.concatenate([np.ones(n_kept_s), np.zeros(n_dropped_s)])
    auroc = float(roc_auc_score(labels, scores))

    # Cohen's d
    mean_k, mean_d = x_kept.mean(), x_dropped.mean()
    std_k,  std_d  = x_kept.std(ddof=1), x_dropped.std(ddof=1)
    pooled_std = np.sqrt(
        ((n_kept_s - 1) * std_k**2 + (n_dropped_s - 1) * std_d**2)
        / (n_kept_s + n_dropped_s - 2)
    )
    cohens_d = float((mean_k - mean_d) / pooled_std) if pooled_std > 0 else float("nan")

    return {
        "n_kept":         n_kept_s,
        "n_dropped":      n_dropped_s,
        "ovl":            ovl,
        "auroc":          auroc,
        "jsd":            jsd,
        "bhattacharyya":  bc,
        "cohens_d":       cohens_d,
        "mean_kept":      float(mean_k),
        "mean_dropped":   float(mean_d),
        "std_kept":       float(std_k),
        "std_dropped":    float(std_d),
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------
def print_metrics(
    label: str, threshold: float,
    metrics: dict, subset_metrics: dict, sep_metrics: dict
):
    print(f"\n{'=' * 68}")
    print(f"  {label}  (threshold = {threshold})")
    print(f"{'=' * 68}")
    print(f"  Total edges : {metrics['n_edges']:,}")

    print(f"\n  --- Bond Length vs Edge Weight (all edges) ---")
    print(f"  Pearson r       : {metrics['pearson_r']:+.4f}  (p = {metrics['pearson_p']:.2e})")
    print(f"  Spearman rho    : {metrics['spearman_rho']:+.4f}  (p = {metrics['spearman_p']:.2e})")
    print(f"  R²  (OLS fit)   : {metrics['r2']:.4f}")
    print(f"  OLS slope       : {metrics['slope']:+.6f}")
    print(f"  OLS intercept   : {metrics['intercept']:+.4f}")

    print(f"\n  --- Distributional Separability (Kept vs Dropped bond lengths) ---")
    if "note" in sep_metrics:
        print(f"  ({sep_metrics['note']})")
    else:
        sm = sep_metrics
        print(f"  Kept  : {sm['n_kept']:>10,}  mean = {sm['mean_kept']:.3f} Å  std = {sm['std_kept']:.3f} Å")
        print(f"  Dropped: {sm['n_dropped']:>9,}  mean = {sm['mean_dropped']:.3f} Å  std = {sm['std_dropped']:.3f} Å")
        print(f"")
        print(f"  Overlap Coeff (OVL)   : {sm['ovl']:.4f}   [0=no overlap → 1=identical]")
        print(f"  AUROC (len→kept/drop) : {sm['auroc']:.4f}   [0.5=random  → 1=perfect sep]")
        print(f"  Jensen-Shannon Div    : {sm['jsd']:.4f}   [0=identical → 1=fully apart]")
        print(f"  Bhattacharyya Coeff   : {sm['bhattacharyya']:.4f}   [0=no overlap → 1=identical]")
        print(f"  Cohen's d             : {sm['cohens_d']:+.4f}  (|d|<0.2 small, 0.5 medium, 0.8 large)")

    for subset in ("kept", "dropped"):
        sm = subset_metrics.get(subset)
        if sm is None:
            continue
        note = f"  [{sm.get('note', '')}]" if "note" in sm else ""
        print(f"\n  --- {subset.capitalize()} edges only ({sm['n_edges']:,}) {note}---")
        if "note" in sm:
            print(f"  (correlation undefined: constant edge-weight values)")
            continue
        print(f"  Pearson r       : {sm['pearson_r']:+.4f}  (p = {sm['pearson_p']:.2e})")
        print(f"  Spearman rho    : {sm['spearman_rho']:+.4f}  (p = {sm['spearman_p']:.2e})")
        print(f"  R²  (OLS fit)   : {sm['r2']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    summary_rows = []

    for exp in EXPERIMENTS:
        label     = exp["label"]
        threshold = exp["threshold"]

        print(f"\nLoading {label} ...")
        weights, lengths, folds_found = load_all_folds(exp)

        if len(weights) == 0:
            print(f"  No data found for {label} — skipping.")
            continue

        print(f"  Loaded {len(weights):,} edges from folds {folds_found}")

        metrics        = compute_correlations(lengths, weights)
        subset_metrics = compute_kept_dropped_correlations(lengths, weights, threshold)
        sep_metrics    = compute_separation_metrics(lengths, weights, threshold)

        print_metrics(label, threshold, metrics, subset_metrics, sep_metrics)

        summary_rows.append({
            "experiment":    label,
            "threshold":     threshold,
            "n_edges":       metrics["n_edges"],
            "pearson_r":     metrics["pearson_r"],
            "spearman_rho":  metrics["spearman_rho"],
            "r2":            metrics["r2"],
            "slope":         metrics["slope"],
            "ovl":           sep_metrics.get("ovl",           float("nan")),
            "auroc":         sep_metrics.get("auroc",         float("nan")),
            "jsd":           sep_metrics.get("jsd",           float("nan")),
            "bhattacharyya": sep_metrics.get("bhattacharyya", float("nan")),
            "cohens_d":      sep_metrics.get("cohens_d",      float("nan")),
        })

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    W = 100
    print(f"\n\n{'=' * W}")
    print("  SUMMARY TABLE  (Bond Length vs Edge Weight, all edges, all folds)")
    print(f"{'=' * W}")
    header = (
        f"{'Experiment':<18} {'θ':>5} {'N edges':>10} "
        f"{'Pearson r':>11} {'Spearman ρ':>12} {'R²':>8} "
        f"{'OVL':>8} {'AUROC':>8} {'JSD':>8} {'BC':>8} {'Cohen d':>10}"
    )
    print(header)
    print("-" * W)
    for row in summary_rows:
        print(
            f"{row['experiment']:<18} "
            f"{row['threshold']:>5.2f} "
            f"{row['n_edges']:>10,} "
            f"{row['pearson_r']:>+11.4f} "
            f"{row['spearman_rho']:>+12.4f} "
            f"{row['r2']:>8.4f} "
            f"{row['ovl']:>8.4f} "
            f"{row['auroc']:>8.4f} "
            f"{row['jsd']:>8.4f} "
            f"{row['bhattacharyya']:>8.4f} "
            f"{row['cohens_d']:>+10.4f}"
        )
    print(f"{'=' * W}")
    print()
    print("  Interpretation guide:")
    print("    OVL   : Overlap Coefficient   — 1.0 = distributions identical, 0 = no overlap")
    print("    AUROC : bond length → kept/drop classifier — 0.5 = random, 1.0 = perfect separation")
    print("    JSD   : Jensen-Shannon Divergence (bits)   — 0 = identical,   1 = fully apart")
    print("    BC    : Bhattacharyya Coefficient           — 1.0 = identical, 0 = no overlap")
    print("    Cohen d: effect size of mean difference     — |d| 0.2/0.5/0.8 = small/medium/large")
    print(f"{'=' * W}")

    # Optional: save summary to CSV
    import csv
    out_csv = os.path.join(os.path.dirname(__file__), "explainability_correlation_metrics.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved to: {out_csv}")


if __name__ == "__main__":
    main()
