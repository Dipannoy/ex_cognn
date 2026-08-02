"""plot_oversmoothing_oversquashing.py

Reads the JSON output from analysis_oversmoothing_oversquashing.py and
produces two publication-quality figures:

  Figure 1 — Oversmoothing  (2 columns: MAE vs depth | MADGap vs depth)
  Figure 2 — Oversquashing  (1 panel: sensitivity vs graph distance,
                              lines per model-type × selected layer depth)

Usage
-----
    # Single property
    python plot_oversmoothing_oversquashing.py \\
        --json_eg outputs/analysis/oversmoothing_oversquashing_mp_eg.json \\
        --json_ef outputs/analysis/oversmoothing_oversquashing_mp_ef.json \\
        --out_dir outputs/analysis/figures

    # Eg only
    python plot_oversmoothing_oversquashing.py \\
        --json_eg outputs/analysis/oversmoothing_oversquashing_mp_eg.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

MANUSCRIPT_RC = {
    "font.family":        "serif",
    "font.size":          13,
    "axes.labelsize":     13,
    "axes.labelweight":   "bold",
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "axes.linewidth":     1.5,
    "xtick.major.width":  1.3,
    "ytick.major.width":  1.3,
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "lines.linewidth":    2.0,
    "lines.markersize":   7,
}

C_BASELINE = "#0072B2"   # deep blue
C_COGNN    = "#E69F00"   # amber
MARKER_BASELINE = "o"
MARKER_COGNN    = "s"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def split_by_model(records: list[dict]) -> tuple[list, list]:
    baseline = sorted([r for r in records if r["model"] == "baseline"],
                      key=lambda r: r["gn_layers"])
    lcsn     = sorted([r for r in records if r["model"] == "lcsn"],
                      key=lambda r: r["gn_layers"])
    return baseline, lcsn


def extract_mae(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layers = np.array([r["gn_layers"] for r in records])
    mae    = np.array([r["test_mae"]     if r["test_mae"]     is not None else np.nan
                       for r in records])
    mae_std= np.array([r["test_mae_std"] if r["test_mae_std"] is not None else np.nan
                       for r in records])
    return layers, mae, mae_std


def extract_mad_gap(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    layers  = np.array([r["gn_layers"] for r in records])
    mad_gap = np.array([r["mad"].get("mad_gap", np.nan) for r in records])
    return layers, mad_gap


def extract_mad_local_remote(records: list[dict]
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layers     = np.array([r["gn_layers"] for r in records])
    mad_local  = np.array([r["mad"].get("mad_local",  np.nan) for r in records])
    mad_remote = np.array([r["mad"].get("mad_remote", np.nan) for r in records])
    return layers, mad_local, mad_remote


def extract_mad_all(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layers      = np.array([r["gn_layers"] for r in records])
    mad_all     = np.array([r["mad"].get("mad_all", np.nan) for r in records])
    mad_all_std = np.array([r["mad"].get("mad_all_std", np.nan) for r in records])
    return layers, mad_all, mad_all_std


def extract_rank_metrics(records: list[dict]
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numerical rank / effective rank vs depth (Zhang et al. 2026, ICLR)."""
    layers      = np.array([r["gn_layers"] for r in records])
    num_rank    = np.array([r.get("rank", {}).get("numerical_rank", np.nan) for r in records])
    num_rank_sd = np.array([r.get("rank", {}).get("numerical_rank_std", np.nan) for r in records])
    eff_rank    = np.array([r.get("rank", {}).get("effective_rank", np.nan) for r in records])
    eff_rank_sd = np.array([r.get("rank", {}).get("effective_rank_std", np.nan) for r in records])
    return layers, num_rank, num_rank_sd, eff_rank, eff_rank_sd


def extract_sensitivity(records: list[dict], max_d: int = 6
                        ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Returns {gn_layers: (mean[1..max_d], std[1..max_d])}."""
    out = {}
    for r in records:
        sens = r.get("sensitivity", {})
        mean = np.array(sens.get("mean", [np.nan] * (max_d + 1)))
        std  = np.array(sens.get("std",  [np.nan] * (max_d + 1)))
        out[r["gn_layers"]] = (mean[1:max_d+1], std[1:max_d+1])
    return out


# ---------------------------------------------------------------------------
# Figure 1 — Oversmoothing (MAE + MADGap vs depth)
# ---------------------------------------------------------------------------

def plot_oversmoothing(ax_mae, ax_mad, baseline, lcsn, prop_label: str):
    """Fill two axes: left=MAE vs layers, right=MADGap vs layers."""

    def _plot(ax, layers, vals, vals_std, color, marker, label):
        ax.errorbar(layers, vals, yerr=vals_std if not np.all(np.isnan(vals_std)) else None,
                    color=color, marker=marker, label=label,
                    capsize=4, capthick=1.5)

    # MAE
    l, m, ms = extract_mae(baseline)
    _plot(ax_mae, l, m, ms, C_BASELINE, MARKER_BASELINE, "Baseline (dense)")
    l, m, ms = extract_mae(lcsn)
    _plot(ax_mae, l, m, ms, C_COGNN, MARKER_COGNN, "CoGNN (pruned)")
    ax_mae.set_xlabel("Number of GCN layers")
    ax_mae.set_ylabel(f"Test MAE (eV)")
    ax_mae.set_title(f"{prop_label} — Test MAE vs depth")
    ax_mae.legend()
    ax_mae.set_xticks(sorted(set(list(extract_mae(baseline)[0]) + list(extract_mae(lcsn)[0]))))

    # MADGap
    l, mg = extract_mad_gap(baseline)
    ax_mad.plot(l, mg, color=C_BASELINE, marker=MARKER_BASELINE,
                label="Baseline (dense)")
    l, mg = extract_mad_gap(lcsn)
    ax_mad.plot(l, mg, color=C_COGNN, marker=MARKER_COGNN,
                label="CoGNN (pruned)")
    ax_mad.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax_mad.set_xlabel("Number of GCN layers")
    ax_mad.set_ylabel("MADGap")
    ax_mad.set_title(f"{prop_label} — MADGap vs depth  (↑ = less oversmoothed)")
    ax_mad.legend()
    ax_mad.set_xticks(sorted(set(list(l))))


# ---------------------------------------------------------------------------
# Figure 1b — MAD_local vs MAD_remote (diagnostic for negative MADGap)
# ---------------------------------------------------------------------------

def plot_mad_local_remote(ax, baseline, lcsn, prop_label: str):
    """Plot MAD_local and MAD_remote vs depth on the same axis.

    MADGap = MAD_remote - MAD_local, so whenever the local curve sits
    *above* the remote curve the gap is negative. Showing both raw curves
    (rather than just the difference) makes it clear whether a negative
    gap comes from local pairs becoming more dissimilar, remote pairs
    becoming more similar, or both.
    """
    l, loc, rem = extract_mad_local_remote(baseline)
    ax.plot(l, loc, color=C_BASELINE, marker=MARKER_BASELINE, linestyle="-",
            label="Baseline — MAD$_{local}$")
    ax.plot(l, rem, color=C_BASELINE, marker=MARKER_BASELINE, linestyle="--",
            label="Baseline — MAD$_{remote}$")

    l, loc, rem = extract_mad_local_remote(lcsn)
    ax.plot(l, loc, color=C_COGNN, marker=MARKER_COGNN, linestyle="-",
            label="CoGNN — MAD$_{local}$")
    ax.plot(l, rem, color=C_COGNN, marker=MARKER_COGNN, linestyle="--",
            label="CoGNN — MAD$_{remote}$")

    ax.set_xlabel("Number of GCN layers")
    ax.set_ylabel("MAD (mean average distance)")
    ax.set_title(f"{prop_label} — MAD$_{{local}}$ vs MAD$_{{remote}}$\n"
                 "(local above remote ⇒ negative MADGap)")
    ax.set_xticks(sorted(set(list(l))))
    ax.legend(fontsize=9, ncol=1)


# ---------------------------------------------------------------------------
# Figure 1c — MAD_all vs depth (sign-agnostic oversmoothing signal)
# ---------------------------------------------------------------------------

def plot_mad_all(ax, baseline, lcsn, prop_label: str):
    """Plot MAD_all (all-pairs mean average distance) vs depth.

    Unlike MADGap, MAD_all does not depend on which of local/remote is
    larger — it collapses toward 0 as depth increases if and only if
    representations are actually becoming less distinguishable overall.
    A model that keeps MAD_all higher at a given depth is retaining more
    representational diversity, regardless of the local/remote sign issue.
    """
    l, mad, mad_std = extract_mad_all(baseline)
    ax.errorbar(l, mad, yerr=mad_std if not np.all(np.isnan(mad_std)) else None,
                color=C_BASELINE, marker=MARKER_BASELINE, label="Baseline (dense)",
                capsize=4, capthick=1.5)
    l, mad, mad_std = extract_mad_all(lcsn)
    ax.errorbar(l, mad, yerr=mad_std if not np.all(np.isnan(mad_std)) else None,
                color=C_COGNN, marker=MARKER_COGNN, label="CoGNN (pruned)",
                capsize=4, capthick=1.5)

    ax.set_xlabel("Number of GCN layers")
    ax.set_ylabel("MAD$_{all}$ (mean average distance)")
    ax.set_title(f"{prop_label} — MAD$_{{all}}$ vs depth  (↑ = less oversmoothed)")
    ax.set_xticks(sorted(set(list(l))))
    ax.legend(fontsize=9)


# ---------------------------------------------------------------------------
# Figure 1d — Rank-based oversmoothing metrics (NumRank & Erank)
#
# Zhang, Deidda, Higham & Tudisco (ICLR 2026) show that MAD/Dirichlet-like
# metrics only signal oversmoothing once they converge exactly to zero,
# which requires very deep, weight-bounded networks — realistic shallow
# GNNs can already suffer performance degradation with no visible MAD
# decay. They propose the numerical rank and effective (Shannon) rank of
# the feature matrix instead, since these are scale-invariant and detect
# convergence toward any low-rank subspace, not just an exact eigenvector
# alignment. Both metrics equal 1 at full rank-1 collapse, so a decreasing
# trend with depth is the oversmoothing signal here (same reading direction
# as MAD_all: higher = less oversmoothed).
# ---------------------------------------------------------------------------

def plot_rank_metrics(ax_num, ax_erank, baseline, lcsn, prop_label: str):
    """Fill two axes: left=NumRank vs depth, right=Erank vs depth."""

    def _plot(ax, layers, vals, vals_std, color, marker, label):
        ax.errorbar(layers, vals, yerr=vals_std if not np.all(np.isnan(vals_std)) else None,
                    color=color, marker=marker, label=label,
                    capsize=4, capthick=1.5)

    l, num, num_sd, _, _ = extract_rank_metrics(baseline)
    _plot(ax_num, l, num, num_sd, C_BASELINE, MARKER_BASELINE, "Baseline (dense)")
    l, num, num_sd, _, _ = extract_rank_metrics(lcsn)
    _plot(ax_num, l, num, num_sd, C_COGNN, MARKER_COGNN, "CoGNN (pruned)")
    ax_num.set_xlabel("Number of GCN layers")
    ax_num.set_ylabel("Numerical Rank")
    ax_num.set_title(f"{prop_label} — Numerical Rank vs depth  (↑ = less oversmoothed)")
    ax_num.set_xticks(sorted(set(list(l))))
    ax_num.legend(fontsize=9)

    l, _, _, erk, erk_sd = extract_rank_metrics(baseline)
    _plot(ax_erank, l, erk, erk_sd, C_BASELINE, MARKER_BASELINE, "Baseline (dense)")
    l, _, _, erk, erk_sd = extract_rank_metrics(lcsn)
    _plot(ax_erank, l, erk, erk_sd, C_COGNN, MARKER_COGNN, "CoGNN (pruned)")
    ax_erank.set_xlabel("Number of GCN layers")
    ax_erank.set_ylabel("Effective Rank")
    ax_erank.set_title(f"{prop_label} — Effective Rank vs depth  (↑ = less oversmoothed)")
    ax_erank.set_xticks(sorted(set(list(l))))
    ax_erank.legend(fontsize=9)


# ---------------------------------------------------------------------------
# Figure 2 — Oversquashing (sensitivity vs distance)
# ---------------------------------------------------------------------------

def plot_oversquashing(ax, baseline, lcsn, prop_label: str,
                       selected_layers: list[int] = None, max_d: int = 6):
    """Plot sensitivity vs distance for selected layer depths."""
    distances = np.arange(1, max_d + 1)

    # choose which layers to show (default: min, mid, max)
    all_layers = sorted(set([r["gn_layers"] for r in baseline + lcsn]))
    if selected_layers is None:
        idxs = [0, len(all_layers)//2, len(all_layers)-1]
        selected_layers = [all_layers[i] for i in idxs]

    sens_base = extract_sensitivity(baseline, max_d)
    sens_lcsn = extract_sensitivity(lcsn, max_d)

    ls_cycle = ["-", "--", ":"]
    for i, n_layers in enumerate(selected_layers):
        ls = ls_cycle[i % len(ls_cycle)]
        if n_layers in sens_base:
            m, s = sens_base[n_layers]
            ax.plot(distances, m, color=C_BASELINE, linestyle=ls,
                    marker=MARKER_BASELINE,
                    label=f"Baseline gn={n_layers}")
            ax.fill_between(distances, m - s, m + s, color=C_BASELINE, alpha=0.12)

        if n_layers in sens_lcsn:
            m, s = sens_lcsn[n_layers]
            ax.plot(distances, m, color=C_COGNN, linestyle=ls,
                    marker=MARKER_COGNN,
                    label=f"CoGNN gn={n_layers}")
            ax.fill_between(distances, m - s, m + s, color=C_COGNN, alpha=0.12)

    ax.set_yscale("log")
    ax.set_xlabel("Graph distance $d(i, j)$")
    ax.set_ylabel(r"Mean $\|\partial\hat{y}/\partial x_j\|_2$")
    ax.set_title(f"{prop_label} — Sensitivity vs distance  (↑ = less oversquashing)")
    ax.set_xticks(distances)
    ax.legend(ncol=2, fontsize=9)


def _label_panels(axes, x: float = -0.14, y: float = 1.06, fontsize: int = 15):
    """Tag each axis in row-major order with (a), (b), (c), ..."""
    for i, ax in enumerate(np.ravel(axes)):
        ax.text(x, y, f"({chr(ord('a') + i)})", transform=ax.transAxes,
                fontsize=fontsize, fontweight="bold", va="bottom", ha="left")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_figures(json_eg: str | None, json_ef: str | None, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    datasets = []
    if json_eg and os.path.exists(json_eg):
        datasets.append(("MP $E_g$", load_json(json_eg)))
    if json_ef and os.path.exists(json_ef):
        datasets.append(("MP $E_f$", load_json(json_ef)))

    if not datasets:
        print("No JSON files found. Provide --json_eg and/or --json_ef.")
        return

    n_props = len(datasets)

    # ── Figure 1: Oversmoothing ────────────────────────────────────────────
    with matplotlib.rc_context(MANUSCRIPT_RC):
        fig1, axes1 = plt.subplots(n_props, 2,
                                   figsize=(12, 4.5 * n_props),
                                   squeeze=False)

        for row, (prop_label, records) in enumerate(datasets):
            baseline, lcsn = split_by_model(records)
            plot_oversmoothing(axes1[row, 0], axes1[row, 1],
                               baseline, lcsn, prop_label)

        # fig1.suptitle("Oversmoothing Analysis: CoGNN vs Baseline",
        #               fontsize=14, fontweight="bold", y=1.01)
        _label_panels(axes1)
        fig1.tight_layout()
        p1 = os.path.join(out_dir, "oversmoothing.pdf")
        fig1.savefig(p1, bbox_inches="tight", dpi=300)
        fig1.savefig(p1.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        plt.close(fig1)
        print(f"Saved {p1}")

    # ── Figure 1b: MAD_local vs MAD_remote (diagnostic) ─────────────────────
    with matplotlib.rc_context(MANUSCRIPT_RC):
        fig1b, axes1b = plt.subplots(1, n_props,
                                     figsize=(6.5 * n_props, 5),
                                     squeeze=False)

        for col, (prop_label, records) in enumerate(datasets):
            baseline, lcsn = split_by_model(records)
            plot_mad_local_remote(axes1b[0, col], baseline, lcsn, prop_label)

        fig1b.tight_layout()
        p1b = os.path.join(out_dir, "mad_local_remote.pdf")
        fig1b.savefig(p1b, bbox_inches="tight", dpi=300)
        fig1b.savefig(p1b.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        plt.close(fig1b)
        print(f"Saved {p1b}")

    # ── Figure 1c: MAD_all vs depth (sign-agnostic oversmoothing signal) ────
    with matplotlib.rc_context(MANUSCRIPT_RC):
        fig1c, axes1c = plt.subplots(1, n_props,
                                     figsize=(6.5 * n_props, 5),
                                     squeeze=False)

        for col, (prop_label, records) in enumerate(datasets):
            baseline, lcsn = split_by_model(records)
            plot_mad_all(axes1c[0, col], baseline, lcsn, prop_label)

        _label_panels(axes1c)
        fig1c.tight_layout()
        p1c = os.path.join(out_dir, "mad_all.pdf")
        fig1c.savefig(p1c, bbox_inches="tight", dpi=300)
        fig1c.savefig(p1c.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        plt.close(fig1c)
        print(f"Saved {p1c}")

    # ── Figure 1d: Rank-based oversmoothing metrics (NumRank & Erank) ───────
    with matplotlib.rc_context(MANUSCRIPT_RC):
        fig1d, axes1d = plt.subplots(n_props, 2,
                                     figsize=(12, 4.5 * n_props),
                                     squeeze=False)

        for row, (prop_label, records) in enumerate(datasets):
            baseline, lcsn = split_by_model(records)
            plot_rank_metrics(axes1d[row, 0], axes1d[row, 1],
                              baseline, lcsn, prop_label)

        _label_panels(axes1d)
        fig1d.tight_layout()
        p1d = os.path.join(out_dir, "rank_metrics.pdf")
        fig1d.savefig(p1d, bbox_inches="tight", dpi=300)
        fig1d.savefig(p1d.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        plt.close(fig1d)
        print(f"Saved {p1d}")

    # ── Figure 2: Oversquashing ────────────────────────────────────────────
    with matplotlib.rc_context(MANUSCRIPT_RC):
        fig2, axes2 = plt.subplots(1, n_props,
                                   figsize=(6.5 * n_props, 5),
                                   squeeze=False)

        for col, (prop_label, records) in enumerate(datasets):
            baseline, lcsn = split_by_model(records)
            plot_oversquashing(axes2[0, col], baseline, lcsn, prop_label)

        # fig2.suptitle("Oversquashing Analysis: Jacobian Sensitivity vs Distance",
        #               fontsize=14, fontweight="bold")
        _label_panels(axes2)
        fig2.tight_layout()
        p2 = os.path.join(out_dir, "oversquashing.pdf")
        fig2.savefig(p2, bbox_inches="tight", dpi=300)
        fig2.savefig(p2.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        plt.close(fig2)
        print(f"Saved {p2}")

    # ── Figure 3: Combined (4-panel) ───────────────────────────────────────
    with matplotlib.rc_context(MANUSCRIPT_RC):
        fig3 = plt.figure(figsize=(14, 10))
        gs   = gridspec.GridSpec(2, 2 * n_props, figure=fig3,
                                 wspace=0.38, hspace=0.45)

        for col_base, (prop_label, records) in enumerate(datasets):
            baseline, lcsn = split_by_model(records)
            ax_mae = fig3.add_subplot(gs[0, col_base * 2])
            ax_mad = fig3.add_subplot(gs[0, col_base * 2 + 1])
            ax_sq  = fig3.add_subplot(gs[1, col_base * 2 : col_base * 2 + 2])

            plot_oversmoothing(ax_mae, ax_mad, baseline, lcsn, prop_label)
            plot_oversquashing(ax_sq,  baseline, lcsn, prop_label)

        fig3.suptitle("CoGNN: Oversmoothing & Oversquashing Analysis (MP20)",
                      fontsize=14, fontweight="bold")
        p3 = os.path.join(out_dir, "oversmoothing_oversquashing_combined.pdf")
        fig3.savefig(p3, bbox_inches="tight", dpi=300)
        fig3.savefig(p3.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        plt.close(fig3)
        print(f"Saved {p3}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot oversmoothing & oversquashing from analysis JSON files"
    )
    p.add_argument("--json_eg", default=None,
                   help="JSON from analysis_oversmoothing_oversquashing.py for Eg")
    p.add_argument("--json_ef", default=None,
                   help="JSON from analysis_oversmoothing_oversquashing.py for Ef")
    p.add_argument("--out_dir", default="outputs/analysis/figures")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_figures(args.json_eg, args.json_ef, args.out_dir)
