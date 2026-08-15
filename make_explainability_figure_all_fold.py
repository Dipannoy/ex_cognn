"""
Compose a manuscript-ready explainability figure (4 × 2 sub-plots).

Layout
------
       Scatter: Edge Weight vs Bond Length | Histogram: Kept vs Dropped
Row 1: JARVIS E_g  (θ=0.3, λ=0.001)
Row 2: JARVIS E_f  (θ=0.5, λ=0.001)
       --- separator ---
Row 3: MP E_g      (θ=0.2, λ=0.001)
Row 4: MP E_f      (θ=0.5, λ=0.001)

Bond weights and lengths are aggregated across all 5 cross-validation folds
(fold 0–4) before plotting.  If the prob_dict JSON files are present in the
results directory the source PNG is regenerated from the raw data with full
manuscript-quality fonts before compositing.  If only the PNG is present it
is used directly.

Usage
-----
    conda run -n cognn python3 make_explainability_figure.py
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
from PIL import Image


# ---------------------------------------------------------------------------
# Paths to the test-set explainability PNGs + JSON keys
# ---------------------------------------------------------------------------
BASE = os.path.join(os.path.dirname(__file__), "downloaded_results/outputs_latest_data")

EXPERIMENTS = [
    {
        "row_label":  r"JARVIS $E_{\rm g}$",
        "theta":      r"$\theta = 0.1$",
        "threshold":  0.1,
        "dir":        os.path.join(BASE, "jarvis_eg_edge_only_scoring_net_th0_1_sw0_001"),
        "json_key":   "test_jarvis_large_eg",
        "png_name":   "explainability_test_jarvis_lg_cell_eg_all_folds.png",
        "group":      "JARVIS",
    },
    {
        "row_label":  r"JARVIS $E_{\rm f}$",
        "theta":      r"$\theta = 0.1$",
        "threshold":  0.1,
        "dir":        os.path.join(BASE, "jarvis_ef_edge_only_scoring_net_th0_1_sw0_001"),
        "json_key":   "test_jarvis_large_ef",
        "png_name":   "explainability_test_jarvis_lg_cell_ef_all_folds.png",
        "group":      "JARVIS",
    },
    {
        "row_label":  r"MP $E_{\rm g}$",
        "theta":      r"$\theta = 0.1$",
        "threshold":  0.1,
        "dir":        os.path.join(BASE, "mp_eg_edge_only_scoring_net_th0_1_sw0_001"),
        "json_key":   "test_mp_large_eg",
        "png_name":   "explainability_test_mp_lg_cell_eg_all_folds.png",
        "group":      "MP",
    },
    {
        "row_label":  r"MP $E_{\rm f}$",
        "theta":      r"$\theta = 0.1$",
        "threshold":  0.1,
        "dir":        os.path.join(BASE, "mp_eg_edge_only_scoring_net_th0_5_sw0_001"),
        "json_key":   "test_mp_large_ef",
        "png_name":   "explainability_test_mp_lg_cell_ef_all_folds.png",
        "group":      "MP",
    },
]

N_FOLDS = 5   # cross-validation folds to aggregate (0 … N_FOLDS-1)

# ---------------------------------------------------------------------------
# Manuscript rcParams (used when regenerating PNGs from JSON)
# ---------------------------------------------------------------------------
MANUSCRIPT_RC = {
    "font.family":       "serif",
    "font.size":         14,
    "axes.labelsize":    14,
    "axes.labelweight":  "bold",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   12,
    "axes.linewidth":    1.8,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.major.size":  5,
    "ytick.major.size":  5,
}

# Okabe-Ito colorblind-safe palette (shared by all plots)
C_KEPT    = "#0072B2"   # deep blue  — retained edges
C_DROPPED = "#E69F00"   # warm amber — pruned edges
C_THRESH  = "#CC3311"   # vivid red  — threshold line


# ---------------------------------------------------------------------------
# Helper: load and aggregate all fold JSON files for one experiment
# ---------------------------------------------------------------------------
def load_all_folds(exp: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Aggregate bond weights and lengths across all N_FOLDS folds.

    Parameters
    ----------
    exp : experiment dict (must contain 'dir', 'json_key')

    Returns
    -------
    all_weights : np.ndarray  [total_edges]
    all_lengths : np.ndarray  [total_edges]
    folds_found : list[int]   which folds were actually loaded
    """
    key     = exp["json_key"]
    out_dir = exp["dir"]

    all_weights: list[float] = []
    all_lengths: list[float] = []
    folds_found: list[int]   = []

    for fold in range(N_FOLDS):
        # Try several common filename patterns produced by runner.py
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

        n_before = len(all_weights)
        for jid, data in prob_data.items():
            weights = data.get("prop_dist", [])
            lengths = data.get("bond_lengths", [])
            if len(weights) == len(lengths) and len(weights) > 0:
                all_weights.extend(weights)
                all_lengths.extend(lengths)

        n_added = len(all_weights) - n_before
        print(f"    [fold {fold}] Added {n_added:,} edges  "
              f"(running total: {len(all_weights):,})")
        folds_found.append(fold)

    return (
        np.array(all_weights, dtype=float),
        np.array(all_lengths, dtype=float),
        folds_found,
    )


# ---------------------------------------------------------------------------
# Helper: regenerate the per-experiment PNG from aggregated fold data
# ---------------------------------------------------------------------------
def regenerate_png_from_json(exp: dict) -> str | None:
    """Regenerate the explainability PNG aggregating all available folds.

    Returns the path of the written PNG, or None if no JSON was found for
    any fold.
    """
    threshold = exp["threshold"]
    out_dir   = exp["dir"]

    print(f"  [{exp['row_label']}] Aggregating folds ...")
    all_weights, all_lengths, folds_found = load_all_folds(exp)

    if len(all_weights) == 0:
        print(f"  WARNING: no bond-length data found for any fold — skipping PNG regeneration.")
        return None

    print(f"  [{exp['row_label']}] Total edges: {len(all_weights):,} "
          f"from folds {folds_found}")

    kept_mask       = all_weights >= threshold
    kept_lengths    = all_lengths[kept_mask]
    dropped_lengths = all_lengths[~kept_mask]

    with matplotlib.rc_context(MANUSCRIPT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ---- Scatter: edge weight vs bond length ----
        axes[0].scatter(
            all_lengths, all_weights,
            alpha=0.12, s=1.5, c=C_KEPT,
            rasterized=True,
        )
        axes[0].axhline(
            y=threshold, color=C_THRESH, linestyle="--",
            linewidth=2.0, label=f"threshold = {threshold}",
        )
        axes[0].legend(loc="upper right")

        # ---- Histogram: kept vs dropped ----
        axes[1].hist(
            kept_lengths, bins=60, alpha=0.65, density=True,
            label=f"Kept ({kept_mask.sum():,})",
            color=C_KEPT, edgecolor="white", linewidth=0.3,
        )
        axes[1].hist(
            dropped_lengths, bins=60, alpha=0.65, density=True,
            label=f"Dropped ({(~kept_mask).sum():,})",
            color=C_DROPPED, edgecolor="white", linewidth=0.3,
        )
        axes[1].legend(loc="upper right")

        plt.tight_layout()
        png_path = os.path.join(out_dir, exp["png_name"])
        plt.savefig(png_path, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"  Saved intermediate PNG: {os.path.basename(png_path)}")

    return png_path


# ---------------------------------------------------------------------------
# Regenerate source PNGs from JSON (all folds) or fall back to existing PNG
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1 — Regenerating per-experiment PNGs from JSON data")
print("=" * 60)

for exp in EXPERIMENTS:
    png_path = os.path.join(exp["dir"], exp["png_name"])
    result   = regenerate_png_from_json(exp)   # JSON always takes priority

    if result:
        print(f"  OK  Regenerated: {os.path.basename(result)}\n")
    elif os.path.exists(png_path):
        print(f"  OK  Using existing PNG: {exp['png_name']}\n")
    else:
        raise FileNotFoundError(
            f"Neither JSON (folds 0–{N_FOLDS - 1}) nor PNG found for "
            f"{exp['row_label']}.\n"
            f"  Expected PNG: {png_path}\n"
            f"  Expected JSON pattern: "
            f"{os.path.join(exp['dir'], 'test_' + exp['json_key'] + '_<fold>_prob_dict.json')}"
        )

# ---------------------------------------------------------------------------
# Global rcParams for the composite manuscript figure
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          11,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

PANEL_LETTERS = list("abcdefgh")


# ---------------------------------------------------------------------------
# Helper: load PNG, strip title bar and white borders, split left/right
# ---------------------------------------------------------------------------
def load_and_split(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load PNG, remove internal title + white borders, split into halves."""
    print(f"  Compositing {os.path.basename(path)} ...")
    img = np.array(Image.open(path).convert("RGB"))
    H, W = img.shape[:2]

    # Strip the matplotlib sub-figure title (occupies top ~40 px)
    TITLE_CROP_TOP = 40
    row_means      = img.mean(axis=(1, 2))
    non_white_rows = np.where(row_means < 248)[0]
    r0 = TITLE_CROP_TOP
    r1 = min(non_white_rows[-1] + 4, H)
    img = img[r0:r1, :, :]

    # Split at horizontal midpoint (white gutter between sub-plots)
    mid   = W // 2
    left  = img[:, :mid, :]
    right = img[:, mid:, :]

    def trim_h(half: np.ndarray) -> np.ndarray:
        """Trim leading/trailing white columns."""
        col_means = half.mean(axis=(0, 2))
        cols = np.where(col_means < 248)[0]
        if len(cols) == 0:
            return half
        return half[:, max(cols[0] - 2, 0):min(cols[-1] + 2, half.shape[1]), :]

    def pad_top(half: np.ndarray, frac: float = 0.07) -> np.ndarray:
        """Add a blank white margin above the plot for the panel letter to sit in."""
        pad_px = max(24, int(round(half.shape[0] * frac)))
        pad    = np.full((pad_px, half.shape[1], 3), 255, dtype=half.dtype)
        return np.vstack([pad, half])

    return pad_top(trim_h(left)), pad_top(trim_h(right))


# ---------------------------------------------------------------------------
# Build composite figure
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 2 — Building composite manuscript figure")
print("=" * 60)

N_ROWS = len(EXPERIMENTS)   # 4
N_COLS = 2                  # scatter | histogram
FIG_W  = 10.0
FIG_H  = 10.5

fig = plt.figure(figsize=(FIG_W, FIG_H))

# Outer gridspec: one row per experiment, with room for group labels on right
# and shared x-/y-axis labels (left margin widened for the merged y-axis
# label; bottom margin widened for the merged x-axis label).
outer = gridspec.GridSpec(
    N_ROWS, 1,
    figure=fig,
    hspace=0.10,
    top=0.95,
    bottom=0.07,
    left=0.17,
    right=0.93,
)

axes: list[list[plt.Axes]] = []
for row_idx, exp in enumerate(EXPERIMENTS):
    inner = gridspec.GridSpecFromSubplotSpec(
        1, N_COLS,
        subplot_spec=outer[row_idx],
        wspace=0.06,
    )
    row_axes = [fig.add_subplot(inner[col]) for col in range(N_COLS)]
    axes.append(row_axes)

# Fill sub-plots
col_titles = ["Edge Weight vs. Bond Length", "Bond Length: Kept vs. Dropped"]

for row_idx, exp in enumerate(EXPERIMENTS):
    left_img, right_img = load_and_split(os.path.join(exp["dir"], exp["png_name"]))

    for col_idx, (ax, im) in enumerate(zip(axes[row_idx], [left_img, right_img])):
        ax.imshow(im, interpolation="lanczos", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Column headers on top row only
        if row_idx == 0:
            ax.set_title(col_titles[col_idx], fontsize=13, fontweight="bold", pad=8)

        # Panel letter (a)–(h) in top-left corner
        letter = PANEL_LETTERS[row_idx * N_COLS + col_idx]
        ax.text(
            0.015, 0.967, f"({letter})",
            transform=ax.transAxes,
            fontsize=13, fontweight="bold",
            va="center", ha="left",
            color="black",
        )

    # Row label on the left  (dataset name + θ)
    axes[row_idx][0].text(
        -0.06, 0.5,
        exp["row_label"] + f"\n{exp['theta']}",
        transform=axes[row_idx][0].transAxes,
        fontsize=12, fontweight="bold",
        va="center", ha="right",
        rotation=0,
        multialignment="center",
    )

# ---------------------------------------------------------------------------
# Horizontal dashed separator between JARVIS block (rows 0–1) and MP (2–3)
# ---------------------------------------------------------------------------
bb1   = axes[1][0].get_position()   # bottom of last JARVIS row
bb2   = axes[2][0].get_position()   # top of first MP row
y_sep = (bb1.y0 + bb2.y1) / 2.0

fig.add_artist(Line2D(
    [0.02, 0.97], [y_sep, y_sep],
    transform=fig.transFigure,
    color="#888888", linewidth=1.2, linestyle="--", clip_on=False,
))

# ---------------------------------------------------------------------------
# Rotated group labels in right margin
# ---------------------------------------------------------------------------
bb_j0    = axes[0][1].get_position()
bb_j1    = axes[1][1].get_position()
y_jarvis = (bb_j0.y1 + bb_j1.y0) / 2.0

bb_m0 = axes[2][1].get_position()
bb_m1 = axes[3][1].get_position()
y_mp  = (bb_m0.y1 + bb_m1.y0) / 2.0

for y_g, label_g in [(y_jarvis, "JARVIS"), (y_mp, "Materials Project")]:
    fig.text(
        0.965, y_g, label_g,
        transform=fig.transFigure,
        fontsize=13, fontweight="bold",
        va="center", ha="center",
        rotation=90,
        color="#222222",
    )

# ---------------------------------------------------------------------------
# Shared axis labels (replace the per-panel labels stripped from the
# individual PNGs): one x-axis label for the whole figure, and one y-axis
# label per column, each spanning all four rows.
# ---------------------------------------------------------------------------
bb_top    = axes[0][0].get_position()
bb_bottom = axes[N_ROWS - 1][0].get_position()
y_center  = (bb_top.y1 + bb_bottom.y0) / 2.0

x_center = (0.17 + 0.93) / 2.0
fig.text(
    x_center, 0.015, "Bond Length (Å)",
    transform=fig.transFigure,
    fontsize=14, fontweight="bold",
    va="bottom", ha="center",
)

label_bbox = dict(boxstyle="square,pad=0.25", fc="white", ec="none")

x_col0_label = 0.045
fig.text(
    x_col0_label, y_center, "Edge Weight",
    transform=fig.transFigure,
    fontsize=13, fontweight="bold",
    va="center", ha="center",
    rotation=90,
    bbox=label_bbox,
    zorder=5,
)

bb_col0 = axes[0][0].get_position()
bb_col1 = axes[0][1].get_position()
x_gutter = (bb_col0.x1 + bb_col1.x0) / 2.0
fig.text(
    x_gutter, y_center, "Density",
    transform=fig.transFigure,
    fontsize=13, fontweight="bold",
    va="center", ha="center",
    rotation=90,
    bbox=label_bbox,
    zorder=5,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_pdf = os.path.join(os.path.dirname(__file__), "explainability_figure_eosn_all_folds_2.pdf")
out_png = os.path.join(os.path.dirname(__file__), "explainability_figure_eosn_all_folds_2.png")

fig.savefig(out_pdf, dpi=300, format="pdf")
fig.savefig(out_png, dpi=300, format="png")
print(f"\nSaved:\n  {out_pdf}\n  {out_png}")