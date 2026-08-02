"""
make_explainability_figure_v2.py

Manuscript-ready redesign of the LearnableCutoffScoringNet explainability
figure. Unlike make_explainability_figure_all_fold.py, this script does NOT
render four standalone PNGs and composite/crop them back together as raster
images. Instead it plots everything natively into a single GridSpec figure,
so axes, ticks, fonts, and labels stay vector (only the dense point cloud is
rasterized), alignment is exact, and axis labels/legends can be merged
cleanly across panels.

Layout
------
       Edge Weight vs. Bond Length (density) | Bond Length: Kept vs. Dropped
Row 1: JARVIS E_g  (theta=0.3)
Row 2: JARVIS E_f  (theta=0.5)
       --- separator ---
Row 3: MP E_g      (theta=0.2)
Row 4: MP E_f      (theta=0.5)

The left column shows a hexbin density plot (edge weight vs. bond length)
instead of a raw alpha-blended scatter of millions of points, which reads
much more cleanly at print resolution and remains far lighter on the vector
PDF. The right column keeps the kept/dropped bond-length histograms.

Usage
-----
    conda run -n cognn python3 make_explainability_figure_v2.py
"""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths to the aggregated per-fold JSON files
# ---------------------------------------------------------------------------
BASE = os.path.join(os.path.dirname(__file__), "downloaded_results/outputs_latest_data")

EXPERIMENTS = [
    {
        "row_label": r"JARVIS $E_{\rm g}$",
        "theta_label": r"$\theta = 0.3$",
        "threshold": 0.3,
        "dir": os.path.join(BASE, "jarvis_eg_learnable_cutoff_scoring_net_th0_3_sw0_001"),
        "json_key": "test_jarvis_large_eg",
    },
    {
        "row_label": r"JARVIS $E_{\rm f}$",
        "theta_label": r"$\theta = 0.5$",
        "threshold": 0.5,
        "dir": os.path.join(BASE, "jarvis_ef_learnable_cutoff_scoring_net_th0_5_sw0_001"),
        "json_key": "test_jarvis_large_ef",
    },
    {
        "row_label": r"MP $E_{\rm g}$",
        "theta_label": r"$\theta = 0.2$",
        "threshold": 0.2,
        "dir": os.path.join(BASE, "mp_eg_learnable_cutoff_scoring_net_th0_2_sw0_001"),
        "json_key": "test_mp_large_eg",
    },
    {
        "row_label": r"MP $E_{\rm f}$",
        "theta_label": r"$\theta = 0.5$",
        "threshold": 0.5,
        "dir": os.path.join(BASE, "mp_ef_learnable_cutoff_scoring_net_th0_5_sw0_001"),
        "json_key": "test_mp_large_ef",
    },
]

N_FOLDS = 5

# Okabe-Ito colorblind-safe palette (unchanged from the original figure)
C_KEPT    = "#0072B2"   # deep blue  — retained edges
C_DROPPED = "#E69F00"   # warm amber — pruned edges
C_THRESH  = "#CC3311"   # vivid red  — threshold line

MANUSCRIPT_RC = {
    "font.family":       "serif",
    "font.size":         12,
    "axes.labelsize":    13,
    "axes.labelweight":  "bold",
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "xtick.labelsize":   10.5,
    "ytick.labelsize":   10.5,
    "legend.fontsize":   10,
    "axes.linewidth":    1.3,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "axes.grid":         True,
    "grid.linestyle":    ":",
    "grid.linewidth":    0.6,
    "grid.alpha":        0.35,
    "figure.dpi":        150,
    "savefig.dpi":       350,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
}


# ---------------------------------------------------------------------------
# Load and aggregate all fold JSON files for one experiment
# ---------------------------------------------------------------------------
def load_all_folds(exp: dict) -> tuple[np.ndarray, np.ndarray]:
    key, out_dir = exp["json_key"], exp["dir"]
    all_weights: list[float] = []
    all_lengths: list[float] = []

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

        print(f"    [fold {fold}] Loading {os.path.basename(json_path)} ...")
        with open(json_path) as f:
            prob_data = json.load(f)

        for jid, data in prob_data.items():
            weights = data.get("prop_dist", [])
            lengths = data.get("bond_lengths", [])
            if len(weights) == len(lengths) and len(weights) > 0:
                all_weights.extend(weights)
                all_lengths.extend(lengths)

    return np.array(all_weights, dtype=float), np.array(all_lengths, dtype=float)


# ---------------------------------------------------------------------------
# Load data for all four experiments up front (needed to fix shared x-limits)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Loading and aggregating fold data")
print("=" * 60)

for exp in EXPERIMENTS:
    print(f"[{exp['row_label']}]")
    w, l = load_all_folds(exp)
    if len(w) == 0:
        raise FileNotFoundError(f"No data found for {exp['row_label']} in {exp['dir']}")
    exp["weights"], exp["lengths"] = w, l
    print(f"  Total edges: {len(w):,}\n")

# Shared x-range (bond length) across every panel, for direct comparability
all_lengths_global = np.concatenate([e["lengths"] for e in EXPERIMENTS])
X_MIN = max(0.0, np.percentile(all_lengths_global, 0.01))
X_MAX = np.percentile(all_lengths_global, 99.95)


# ---------------------------------------------------------------------------
# Build the figure
# ---------------------------------------------------------------------------
print("=" * 60)
print("Building composite figure")
print("=" * 60)

N_COLS = 2
PANEL_LETTERS = list("abcdefgh")

# Map each experiment onto a 5-row grid: rows 0-1 = JARVIS, row 2 = a
# dedicated (blank) spacer, rows 3-4 = Materials Project. Giving the
# JARVIS/MP transition its own reserved row -- rather than relying on the
# same hspace used between every pair of rows -- keeps the separator line,
# group labels, and merged axis labels clear of the panel letters above
# rows 3-4.
GRID_ROW_OF_EXP = [0, 1, 3, 4]
SPACER_GRID_ROW = 2

with matplotlib.rc_context(MANUSCRIPT_RC):
    fig = plt.figure(figsize=(11.0, 13.0))

    outer = gridspec.GridSpec(
        5, 1, figure=fig,
        height_ratios=[1, 1, 0.18, 1, 1],
        hspace=0.38,
        top=0.90, bottom=0.075, left=0.19, right=0.90,
    )

    spacer_ax = fig.add_subplot(outer[SPACER_GRID_ROW])
    spacer_ax.axis("off")

    axes: list[list[plt.Axes]] = []
    for exp_idx in range(len(EXPERIMENTS)):
        grid_row = GRID_ROW_OF_EXP[exp_idx]
        inner = gridspec.GridSpecFromSubplotSpec(
            1, N_COLS, subplot_spec=outer[grid_row], wspace=0.16,
        )
        axes.append([fig.add_subplot(inner[c]) for c in range(N_COLS)])

    col_titles = ["Edge Weight vs. Bond Length", "Bond Length: Kept vs. Dropped"]

    for row_idx, exp in enumerate(EXPERIMENTS):
        weights, lengths = exp["weights"], exp["lengths"]
        threshold = exp["threshold"]
        kept_mask = weights >= threshold

        ax_scatter, ax_hist = axes[row_idx]

        # ---- Left: edge weight vs. bond length scatter ----
        # Same scatter style as make_explainability_figure_all_fold.py
        # (plain alpha-blended points in C_KEPT, rasterized for a light PDF).
        ax_scatter.scatter(
            lengths, weights,
            alpha=0.12, s=1.5, c=C_KEPT, rasterized=True,
        )
        thresh_line = ax_scatter.axhline(
            y=threshold, color=C_THRESH, linestyle="--", linewidth=1.8, zorder=5,
        )
        ax_scatter.set_xlim(X_MIN, X_MAX)
        ax_scatter.set_ylim(-0.03, 1.03)
        # Threshold as a legend entry (not an inline tag on the line itself,
        # which got clipped whenever theta sat near the top/bottom edge).
        ax_scatter.legend(
            handles=[thresh_line], labels=[exp["theta_label"]],
            loc="upper right", frameon=True, framealpha=0.85,
            fontsize=9, handlelength=1.5, borderpad=0.4,
        )

        # ---- Right: kept vs. dropped bond-length histograms ----
        bins = np.linspace(X_MIN, X_MAX, 70)
        ax_hist.hist(
            lengths[kept_mask], bins=bins, density=True,
            alpha=0.68, color=C_KEPT, edgecolor="white", linewidth=0.3,
            label=f"Kept ({kept_mask.sum():,})",
        )
        ax_hist.hist(
            lengths[~kept_mask], bins=bins, density=True,
            alpha=0.68, color=C_DROPPED, edgecolor="white", linewidth=0.3,
            label=f"Dropped ({(~kept_mask).sum():,})",
        )
        ax_hist.set_xlim(X_MIN, X_MAX)
        ax_hist.legend(
            loc="upper left", frameon=True, framealpha=0.85,
            fontsize=8.5, handlelength=1.2, borderpad=0.4,
        )

        for ax in (ax_scatter, ax_hist):
            ax.set_axisbelow(True)

        # Panel letters, placed fully outside/above each axes so they never
        # sit over the plotted content. Row 0 gets extra clearance because
        # of the column titles above it.
        letter_y = 1.34 if row_idx == 0 else 1.08
        title_pad = 10 if row_idx == 0 else None
        if row_idx == 0:
            ax_scatter.set_title(col_titles[0], pad=title_pad)
            ax_hist.set_title(col_titles[1], pad=title_pad)

        for col_idx, ax in enumerate((ax_scatter, ax_hist)):
            letter = PANEL_LETTERS[row_idx * N_COLS + col_idx]
            ax.text(
                -0.16, letter_y, f"({letter})",
                transform=ax.transAxes,
                fontsize=14, fontweight="bold",
                va="bottom", ha="left", color="black",
            )

        # Row label (dataset + theta), left of the density panel
        ax_scatter.text(
            -0.45, 0.5, exp["row_label"] + "\n" + exp["theta_label"],
            transform=ax_scatter.transAxes,
            fontsize=12.5, fontweight="bold",
            va="center", ha="center", multialignment="center",
        )

    # -----------------------------------------------------------------
    # Dashed separator + group labels, centered on the dedicated spacer row
    # -----------------------------------------------------------------
    bb_spacer = spacer_ax.get_position()
    y_sep = (bb_spacer.y0 + bb_spacer.y1) / 2.0
    fig.add_artist(Line2D(
        [0.04, 0.985], [y_sep, y_sep], transform=fig.transFigure,
        color="#999999", linewidth=1.1, linestyle="--", clip_on=False,
    ))

    bb_j0, bb_j1 = axes[0][1].get_position(), axes[1][1].get_position()
    bb_m0, bb_m1 = axes[2][1].get_position(), axes[3][1].get_position()
    for y_g, label_g in [
        ((bb_j0.y1 + bb_j1.y0) / 2.0, "JARVIS"),
        ((bb_m0.y1 + bb_m1.y0) / 2.0, "Materials Project"),
    ]:
        fig.text(
            0.975, y_g, label_g, transform=fig.transFigure,
            fontsize=13, fontweight="bold", va="center", ha="center",
            rotation=90, color="#222222",
        )

    # -----------------------------------------------------------------
    # Merged shared axis labels (one x-label, one y-label per column),
    # centered on the whole figure and vertically parked in the spacer row
    # so they never compete with a panel letter for space.
    # -----------------------------------------------------------------
    label_bbox = dict(boxstyle="square,pad=0.25", fc="white", ec="none")

    fig.text(
        (0.19 + 0.90) / 2.0, 0.02, "Bond Length (Å)",
        transform=fig.transFigure, fontsize=14, fontweight="bold",
        va="bottom", ha="center",
    )
    fig.text(
        0.05, y_sep, "Edge Weight", transform=fig.transFigure,
        fontsize=13, fontweight="bold", va="center", ha="center",
        rotation=90, bbox=label_bbox, zorder=5,
    )
    bb_col0, bb_col1 = axes[0][0].get_position(), axes[0][1].get_position()
    fig.text(
        (bb_col0.x1 + bb_col1.x0) / 2.0, y_sep, "Density",
        transform=fig.transFigure, fontsize=13, fontweight="bold",
        va="center", ha="center", rotation=90, bbox=label_bbox, zorder=5,
    )

    out_pdf = os.path.join(os.path.dirname(__file__), "explainability_figure_v2.pdf")
    out_png = os.path.join(os.path.dirname(__file__), "explainability_figure_v2.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)

print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
