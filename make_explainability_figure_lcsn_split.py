"""
make_explainability_figure_lcsn_split.py

LearnableCutoffScoringNet explainability figure, split into TWO figures
(one per dataset) instead of the single 4x2 = 8-panel composite.

Why this differs from make_explainability_figure_v2.py
------------------------------------------------------
* Two output figures, 2x2 panels each -- JARVIS and Materials Project --
  rather than one 4x2 figure. Eight panels in a single float forced a tall
  aspect that left large empty bands above and below the axes once LaTeX
  scaled it to \textwidth.
* Row labels are now just "E_g" / "E_f". In the combined figure each row
  had to carry its dataset name too, which is what pushed the labels far
  out into the left margin (left=0.19, and text at x=-0.45 in axes
  coords). With one dataset per figure the dataset name belongs in the
  caption, so the labels are short and sit next to the axes.
* constrained_layout replaces the hand-tuned GridSpec margins, the 0.18
  spacer row, and the figure-level separator line, all of which existed
  only to keep the two dataset groups apart.

Usage
-----
    conda run -n cognn python3 make_explainability_figure_lcsn_split.py
"""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), "downloaded_results/outputs_latest_data")

# One entry per output figure.
FIGURES = [
    {
        "dataset": "JARVIS",
        "slug": "jarvis",
        "rows": [
            {
                "row_label": r"$E_{\rm g}$",
                "theta_label": r"$\theta = 0.3$",
                "threshold": 0.3,
                "dir": os.path.join(BASE, "jarvis_eg_learnable_cutoff_scoring_net_th0_3_sw0_001"),
                "json_key": "test_jarvis_large_eg",
            },
            {
                "row_label": r"$E_{\rm f}$",
                "theta_label": r"$\theta = 0.5$",
                "threshold": 0.5,
                "dir": os.path.join(BASE, "jarvis_ef_learnable_cutoff_scoring_net_th0_5_sw0_001"),
                "json_key": "test_jarvis_large_ef",
            },
        ],
    },
    {
        "dataset": "Materials Project",
        "slug": "mp",
        "rows": [
            {
                "row_label": r"$E_{\rm g}$",
                "theta_label": r"$\theta = 0.2$",
                "threshold": 0.2,
                "dir": os.path.join(BASE, "mp_eg_learnable_cutoff_scoring_net_th0_2_sw0_001"),
                "json_key": "test_mp_large_eg",
            },
            {
                "row_label": r"$E_{\rm f}$",
                "theta_label": r"$\theta = 0.5$",
                "threshold": 0.5,
                "dir": os.path.join(BASE, "mp_ef_learnable_cutoff_scoring_net_th0_5_sw0_001"),
                "json_key": "test_mp_large_ef",
            },
        ],
    },
]

N_FOLDS = 5

# Okabe-Ito colorblind-safe palette (unchanged)
C_KEPT    = "#0072B2"   # deep blue  -- retained edges
C_DROPPED = "#E69F00"   # warm amber -- pruned edges
C_THRESH  = "#CC3311"   # vivid red  -- threshold line

MANUSCRIPT_RC = {
    "font.family":       "serif",
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.labelweight":  "bold",
    "axes.titlesize":    12.5,
    "axes.titleweight":  "bold",
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9.5,
    "axes.linewidth":    1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "axes.grid":         True,
    "grid.linestyle":    ":",
    "grid.linewidth":    0.6,
    "grid.alpha":        0.35,
    "figure.dpi":        150,
    "savefig.dpi":       350,
    "savefig.pad_inches": 0.02,
}


def load_all_folds(row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate edge weights and bond lengths over all CV folds."""
    key, out_dir = row["json_key"], row["dir"]
    all_weights: list[float] = []
    all_lengths: list[float] = []

    for fold in range(N_FOLDS):
        candidates = [
            os.path.join(out_dir, f"{key}_{fold}_prob_dict.json"),
            os.path.join(out_dir, f"test_{key}_{fold}_prob_dict.json"),
            os.path.join(out_dir, f"{key}_test_{fold}_prob_dict.json"),
        ]
        json_path = next((p for p in candidates if os.path.exists(p)), None)
        if json_path is None:
            print(f"    [fold {fold}] WARNING: no JSON found -- skipping.")
            continue

        print(f"    [fold {fold}] Loading {os.path.basename(json_path)} ...")
        with open(json_path) as f:
            prob_data = json.load(f)

        for _jid, data in prob_data.items():
            weights = data.get("prop_dist", [])
            lengths = data.get("bond_lengths", [])
            if len(weights) == len(lengths) and len(weights) > 0:
                all_weights.extend(weights)
                all_lengths.extend(lengths)

    return np.array(all_weights, dtype=float), np.array(all_lengths, dtype=float)


# ---------------------------------------------------------------------------
# Load everything first so both figures can share one bond-length x-range
# ---------------------------------------------------------------------------
print("=" * 60)
print("Loading and aggregating fold data")
print("=" * 60)

for figspec in FIGURES:
    for row in figspec["rows"]:
        print(f"[{figspec['dataset']} {row['row_label']}]")
        w, l = load_all_folds(row)
        if len(w) == 0:
            raise FileNotFoundError(f"No data found in {row['dir']}")
        row["weights"], row["lengths"] = w, l
        print(f"  Total edges: {len(w):,}\n")

all_lengths_global = np.concatenate(
    [r["lengths"] for f in FIGURES for r in f["rows"]]
)
X_MIN = max(0.0, np.percentile(all_lengths_global, 0.01))
X_MAX = np.percentile(all_lengths_global, 99.95)
print(f"Shared bond-length range: {X_MIN:.2f} - {X_MAX:.2f} A\n")


# ---------------------------------------------------------------------------
# Build one figure per dataset
# ---------------------------------------------------------------------------
COL_TITLES = ["Edge Weight vs. Bond Length", "Bond Length: Kept vs. Dropped"]
PANEL_LETTERS = list("abcd")

with matplotlib.rc_context(MANUSCRIPT_RC):
    for figspec in FIGURES:
        print("=" * 60)
        print(f"Building figure: {figspec['dataset']}")
        print("=" * 60)

        rows = figspec["rows"]

        # One sub-figure per column, so each column can carry a single shared
        # y-label spanning both of its rows ("Edge Weight" / "Density") via
        # supylabel, instead of repeating the label on every axes.
        fig = plt.figure(figsize=(9.4, 5.4), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.0, hspace=0.0)
        subfigs = fig.subfigures(1, 2, wspace=0.02)

        axes = [[None, None] for _ in rows]
        for col_idx in range(2):
            col_axes = subfigs[col_idx].subplots(len(rows), 1, sharex=True)
            # Small h_pad pulls the column heading down towards its axes; the
            # gap it leaves is then taken up by the panel letter's title pad.
            subfigs[col_idx].get_layout_engine().set(h_pad=0.005, hspace=0.05)
            for r in range(len(rows)):
                axes[r][col_idx] = col_axes[r]

        subfigs[0].supylabel("Edge Weight", fontweight="bold", fontsize=11)
        subfigs[1].supylabel("Density", fontweight="bold", fontsize=11)
        fig.supxlabel("Bond Length (Å)", fontweight="bold", fontsize=11)

        # The column heading is a centred axes title on the top row, sharing
        # its baseline with the left-aligned panel letter (see below). As a
        # sub-figure suptitle it sat a whole text row higher and could only be
        # lowered via h_pad; constrained_layout overrides an explicit y and
        # then expands the axes into the space freed, putting the heading on
        # top of the plot. On one baseline, `pad` controls the gap directly.

        for row_idx, row in enumerate(rows):
            weights, lengths = row["weights"], row["lengths"]
            threshold = row["threshold"]
            kept_mask = weights >= threshold

            ax_scatter, ax_hist = axes[row_idx]

            # ---- Left: edge weight vs. bond length ----
            ax_scatter.scatter(
                lengths, weights,
                alpha=0.12, s=1.5, c=C_KEPT, rasterized=True,
            )
            thresh_line = ax_scatter.axhline(
                y=threshold, color=C_THRESH, linestyle="--", linewidth=1.8, zorder=5,
            )
            ax_scatter.set_xlim(X_MIN, X_MAX)
            ax_scatter.set_ylim(-0.03, 1.03)
            ax_scatter.legend(
                handles=[thresh_line], labels=[row["theta_label"]],
                loc="upper right", frameon=True, framealpha=0.85,
                handlelength=1.5, borderpad=0.4,
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

            for col_idx, ax in enumerate((ax_scatter, ax_hist)):
                ax.set_axisbelow(True)
                # Panel letter as a left-aligned axes title. Free-floating
                # text inside the axes collided with the scatter cloud and
                # hid under the histogram legend; as a title, the layout
                # engine reserves space for it.
                ax.set_title(
                    f"({PANEL_LETTERS[row_idx * 2 + col_idx]})",
                    loc="left", fontsize=11.5, fontweight="bold", pad=9,
                )
                if row_idx == 0:
                    ax.set_title(
                        COL_TITLES[col_idx],
                        loc="center", fontsize=11.5, fontweight="bold", pad=9,
                    )

            # Row label on the far right. The left margin now belongs to the
            # shared "Edge Weight" label, and using a right-side y-label (as
            # opposed to free-floating text) means constrained_layout reserves
            # room for it instead of clipping it at the figure edge.
            ax_hist.yaxis.set_label_position("right")
            ax_hist.set_ylabel(
                row["row_label"], rotation=90, labelpad=10, fontsize=13,
            )

        slug = figspec["slug"]
        out_pdf = os.path.join(
            os.path.dirname(__file__), f"explainability_figure_lcsn_{slug}.pdf")
        out_png = os.path.join(
            os.path.dirname(__file__), f"explainability_figure_lcsn_{slug}.png")
        fig.savefig(out_pdf)
        fig.savefig(out_png)
        plt.close(fig)
        print(f"Saved:\n  {out_pdf}\n  {out_png}\n")
