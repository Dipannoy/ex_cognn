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
from matplotlib.colors import LinearSegmentedColormap, to_hex

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


def extract_mad_gap(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layers      = np.array([r["gn_layers"] for r in records])
    mad_gap     = np.array([r["mad"].get("mad_gap", np.nan) for r in records])
    mad_gap_std = np.array([r["mad"].get("mad_gap_std", np.nan) for r in records])
    return layers, mad_gap, mad_gap_std


def extract_mad_local_remote(records: list[dict]
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    layers         = np.array([r["gn_layers"] for r in records])
    mad_local      = np.array([r["mad"].get("mad_local",  np.nan) for r in records])
    mad_local_std  = np.array([r["mad"].get("mad_local_std",  np.nan) for r in records])
    mad_remote     = np.array([r["mad"].get("mad_remote", np.nan) for r in records])
    mad_remote_std = np.array([r["mad"].get("mad_remote_std", np.nan) for r in records])
    return layers, mad_local, mad_local_std, mad_remote, mad_remote_std


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

def _mae_unit(prop_label: str) -> str:
    """Band gap (E_g) is reported in eV; formation energy (E_f) is reported
    per atom, eV/atom (see data_prep_generalizability.py)."""
    return "eV/atom" if "E_f" in prop_label else "eV"


def plot_oversmoothing(ax_mae, ax_mad, baseline, lcsn, prop_label: str):
    """Fill two axes: left=MAE vs layers, right=MADGap vs layers."""

    def _plot(ax, layers, vals, vals_std, color, marker, label):
        has_err = not np.all(np.isnan(vals_std))
        ax.errorbar(layers, vals, yerr=vals_std if has_err else None,
                    color=color, marker=marker, label=label,
                    ecolor=_lighten(color) if has_err else color,
                    elinewidth=1.2, capsize=4, capthick=1.2)

    # MAE
    l, m, ms = extract_mae(baseline)
    _plot(ax_mae, l, m, ms, C_BASELINE, MARKER_BASELINE, "Baseline (dense)")
    l, m, ms = extract_mae(lcsn)
    _plot(ax_mae, l, m, ms, C_COGNN, MARKER_COGNN, "CoGNN (pruned)")
    ax_mae.set_xlabel("Number of GCN layers")
    ax_mae.set_ylabel(f"Test MAE ({_mae_unit(prop_label)})")
    ax_mae.set_title(f"{prop_label} — Test MAE vs depth")
    ax_mae.legend()
    ax_mae.set_xticks(sorted(set(list(extract_mae(baseline)[0]) + list(extract_mae(lcsn)[0]))))

    # MADGap
    l, mg, mg_std = extract_mad_gap(baseline)
    _plot(ax_mad, l, mg, mg_std, C_BASELINE, MARKER_BASELINE, "Baseline (dense)")
    l, mg, mg_std = extract_mad_gap(lcsn)
    _plot(ax_mad, l, mg, mg_std, C_COGNN, MARKER_COGNN, "CoGNN (pruned)")
    ax_mad.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax_mad.set_xlabel("Number of GCN layers")
    ax_mad.set_ylabel("MADGap")
    ax_mad.set_title(f"{prop_label} — MADGap vs depth  (↑ = less oversmoothed)")
    ax_mad.legend()
    ax_mad.set_xticks(sorted(set(list(l))))


# ---------------------------------------------------------------------------
# Figure 1b — MAD_local vs MAD_remote (diagnostic for negative MADGap)
# ---------------------------------------------------------------------------

_MAD_LOCAL_REMOTE_STYLE = {
    ("baseline", "local"):  dict(color="#0072B2", marker="o"),   # blue
    ("baseline", "remote"): dict(color="#56B4E9", marker="o"),   # sky blue
    ("cognn",    "local"):  dict(color="#E69F00", marker="s"),   # orange
    ("cognn",    "remote"): dict(color="#D55E00", marker="s"),   # vermillion
}


def plot_mad_local_remote(ax, baseline, lcsn, prop_label: str):
    """Plot MAD_local and MAD_remote vs depth on the same axis.

    MADGap = MAD_remote - MAD_local, so whenever the local curve sits
    *above* the remote curve the gap is negative. Showing both raw curves
    (rather than just the difference) makes it clear whether a negative
    gap comes from local pairs becoming more dissimilar, remote pairs
    becoming more similar, or both. Local/remote is now encoded by color
    (not linestyle) — four distinct colors, one per (model, local/remote).
    """

    def _plot(layers, vals, vals_std, model_key, part_key, label):
        has_err = not np.all(np.isnan(vals_std))
        style = _MAD_LOCAL_REMOTE_STYLE[(model_key, part_key)]
        ax.errorbar(layers, vals, yerr=vals_std if has_err else None,
                    label=label,
                    ecolor=_lighten(style["color"]) if has_err else style["color"],
                    elinewidth=1.2, capsize=4, capthick=1.2,
                    **style)

    l, loc, loc_sd, rem, rem_sd = extract_mad_local_remote(baseline)
    _plot(l, loc, loc_sd, "baseline", "local",  "Baseline — MAD$_{local}$")
    _plot(l, rem, rem_sd, "baseline", "remote", "Baseline — MAD$_{remote}$")

    l, loc, loc_sd, rem, rem_sd = extract_mad_local_remote(lcsn)
    _plot(l, loc, loc_sd, "cognn", "local",  "CoGNN — MAD$_{local}$")
    _plot(l, rem, rem_sd, "cognn", "remote", "CoGNN — MAD$_{remote}$")

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
    has_err = not np.all(np.isnan(mad_std))
    ax.errorbar(l, mad, yerr=mad_std if has_err else None,
                color=C_BASELINE, marker=MARKER_BASELINE, label="Baseline (dense)",
                ecolor=_lighten(C_BASELINE) if has_err else C_BASELINE,
                elinewidth=1.2, capsize=4, capthick=1.2)
    l, mad, mad_std = extract_mad_all(lcsn)
    has_err = not np.all(np.isnan(mad_std))
    ax.errorbar(l, mad, yerr=mad_std if has_err else None,
                color=C_COGNN, marker=MARKER_COGNN, label="CoGNN (pruned)",
                ecolor=_lighten(C_COGNN) if has_err else C_COGNN,
                elinewidth=1.2, capsize=4, capthick=1.2)

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
        has_err = not np.all(np.isnan(vals_std))
        ax.errorbar(layers, vals, yerr=vals_std if has_err else None,
                    color=color, marker=marker, label=label,
                    ecolor=_lighten(color) if has_err else color,
                    elinewidth=1.2, capsize=4, capthick=1.2)

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
# Figure 1e — MAE panel | Rank panel (NumRank + Erank on a twin axis), side
# by side, one row per property. Real units throughout — no normalization.
# Each panel's caption is prefixed with the property label
# ("MP $E_g$ — MAE vs GCN layers" / "MP $E_g$ — Rank vs GCN layers").
# ---------------------------------------------------------------------------

def _lighten(hex_color: str, factor: float = 0.55) -> str:
    """Blend a hex color toward white (factor=0 -> unchanged, 1 -> white)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


_METRIC_LABEL = {"numrank": "NumRank", "erank": "Erank"}
_MODEL_LABEL  = {"baseline": "Baseline", "cognn": "CoGNN"}

# Four distinct colors for the rank panel's two metrics x two models — cool
# hues for baseline, warm hues for CoGNN (same convention as everywhere
# else); marker shape is a secondary cue for which metric a line is.
_RANK_SERIES_STYLE = {
    ("baseline", "numrank"): dict(color="#009E73", marker="s"),   # green
    ("cognn",    "numrank"): dict(color="#D55E00", marker="s"),   # vermillion
    ("baseline", "erank"):   dict(color="#56B4E9", marker="^"),   # sky blue
    ("cognn",    "erank"):   dict(color="#CC79A7", marker="^"),   # purple
}


def plot_mae_panel(ax, baseline, lcsn, prop_label: str):
    """MAE vs depth, real units, with error bars."""

    def _plot(layers, vals, vals_std, color, marker, label):
        has_err = not np.all(np.isnan(vals_std))
        ax.errorbar(layers, vals, yerr=vals_std if has_err else None,
                    color=color, marker=marker, label=label,
                    ecolor=_lighten(color) if has_err else color,
                    elinewidth=1.2, capsize=4, capthick=1.2)

    l, m, ms = extract_mae(baseline)
    _plot(l, m, ms, C_BASELINE, MARKER_BASELINE, "Baseline (dense)")
    l, m, ms = extract_mae(lcsn)
    _plot(l, m, ms, C_COGNN, MARKER_COGNN, "CoGNN (pruned)")
    ax.set_xlabel("Number of GCN layers")
    ax.set_ylabel(f"{prop_label} — Test MAE ({_mae_unit(prop_label)})")
    ax.set_title(f"{prop_label} — MAE vs GCN layers")
    ax.set_xticks(sorted(set(l.tolist())))
    ax.legend(fontsize=9)


def plot_rank_panel(ax, baseline, lcsn, prop_label: str):
    """NumRank (left axis) + Erank (right axis) vs depth, real units."""
    ax_num = ax
    ax_erk = ax.twinx()

    def _plot(ax_, layers, vals, vals_std, model_key, metric_key):
        has_err = not np.all(np.isnan(vals_std))
        style = _RANK_SERIES_STYLE[(model_key, metric_key)]
        label = f"{_MODEL_LABEL[model_key]} — {_METRIC_LABEL[metric_key]}"
        ax_.errorbar(layers, vals, yerr=vals_std if has_err else None,
                     linestyle="-", markersize=6.5, label=label,
                     ecolor=_lighten(style["color"]) if has_err else style["color"],
                     elinewidth=1.0, capsize=3, capthick=1.0,
                     **style)

    l_b, num_b, num_b_sd, erk_b, erk_b_sd = extract_rank_metrics(baseline)
    l_c, num_c, num_c_sd, erk_c, erk_c_sd = extract_rank_metrics(lcsn)

    _plot(ax_num, l_b, num_b, num_b_sd, "baseline", "numrank")
    _plot(ax_num, l_c, num_c, num_c_sd, "cognn",    "numrank")
    _plot(ax_erk, l_b, erk_b, erk_b_sd, "baseline", "erank")
    _plot(ax_erk, l_c, erk_c, erk_c_sd, "cognn",    "erank")

    ax_num.set_xlabel("Number of GCN layers")
    ax_num.set_ylabel("Numerical Rank")
    ax_erk.set_ylabel("Effective Rank")
    ax_num.set_title(f"{prop_label} — Rank vs GCN layers")
    ax_num.set_xticks(sorted(set(l_b.tolist()) | set(l_c.tolist())))

    h1, lab1 = ax_num.get_legend_handles_labels()
    h2, lab2 = ax_erk.get_legend_handles_labels()
    ax_num.legend(h1 + h2, lab1 + lab2, fontsize=8, ncol=2, loc="best")


# ---------------------------------------------------------------------------
# Figure 2 — Oversquashing (sensitivity vs distance)
# ---------------------------------------------------------------------------

# Custom-anchored colormaps for depth shading -- built from hand-picked,
# well-saturated endpoints rather than matplotlib's stock "Blues"/"Oranges"
# (which are washed out at low sample fractions and read poorly in print).
# Light -> dark still maps to shallow -> deep network, within each model's
# hue family.
_OVERSQ_BASE_CMAP  = LinearSegmentedColormap.from_list(
    "oversq_base",  ["#8FCBEB", "#0072B2", "#00263D"])
_OVERSQ_COGNN_CMAP = LinearSegmentedColormap.from_list(
    "oversq_cognn", ["#FBCB8B", "#E69F00", "#7F2704"])


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

    # Depth is encoded by color shade (not linestyle): each selected depth
    # gets its own color from the custom colormaps above. Error bars (not
    # translucent fill_between bands, which muddy together wherever a
    # baseline and CoGNN band overlap) match the style used in every other
    # panel in this file.
    n = max(len(selected_layers), 1)
    shades = np.linspace(0.12, 0.95, n)
    base_colors  = [to_hex(_OVERSQ_BASE_CMAP(s))  for s in shades]
    cognn_colors = [to_hex(_OVERSQ_COGNN_CMAP(s)) for s in shades]

    for i, n_layers in enumerate(selected_layers):
        if n_layers in sens_base:
            m, s = sens_base[n_layers]
            has_err = np.any(s > 0)
            ax.errorbar(distances, m, yerr=s if has_err else None,
                        color=base_colors[i], marker=MARKER_BASELINE,
                        ecolor=_lighten(base_colors[i]) if has_err else base_colors[i],
                        elinewidth=1.1, capsize=3, capthick=1.1,
                        label=f"Baseline gn={n_layers}")

        if n_layers in sens_lcsn:
            m, s = sens_lcsn[n_layers]
            has_err = np.any(s > 0)
            ax.errorbar(distances, m, yerr=s if has_err else None,
                        color=cognn_colors[i], marker=MARKER_COGNN,
                        ecolor=_lighten(cognn_colors[i]) if has_err else cognn_colors[i],
                        elinewidth=1.1, capsize=3, capthick=1.1,
                        label=f"CoGNN gn={n_layers}")

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

    # ── Figure 1e: MAE panel | Rank panel, side by side, one row per property ──
    with matplotlib.rc_context(MANUSCRIPT_RC):
        fig1e, axes1e = plt.subplots(n_props, 2,
                                     figsize=(13.0, 5.0 * n_props),
                                     squeeze=False)

        for row, (prop_label, records) in enumerate(datasets):
            baseline, lcsn = split_by_model(records)
            plot_mae_panel(axes1e[row, 0], baseline, lcsn, prop_label)
            plot_rank_panel(axes1e[row, 1], baseline, lcsn, prop_label)

        _label_panels(axes1e)
        fig1e.tight_layout()
        p1e = os.path.join(out_dir, "rank_mae_combined.pdf")
        fig1e.savefig(p1e, bbox_inches="tight", dpi=300)
        fig1e.savefig(p1e.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        plt.close(fig1e)
        print(f"Saved {p1e}")

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
