"""
Compose a manuscript-ready explainability figure (4 × 2 sub-plots).

Layout
------
       Scatter: Edge Weight vs Bond Length | Histogram: Kept vs Dropped
Row 1: JARVIS E_g  (θ=0.4, λ=0.001)
Row 2: JARVIS E_f  (θ=0.4, λ=0.001)
       --- separator ---
Row 3: MP E_g      (θ=0.2, λ=0.001)
Row 4: MP E_f      (θ=0.5, λ=0.001)

If the prob_dict JSON file for an experiment is present in its results directory,
the source PNG is regenerated from the raw data with full manuscript-quality fonts
before compositing.  If only the PNG is present, it is used directly.

Usage
-----
    conda run -n cognn python3 make_explainability_figure.py
"""

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
# Paths to the test-set fold-0 explainability PNGs + JSON keys
# ---------------------------------------------------------------------------
BASE = os.path.join(os.path.dirname(__file__), "downloaded_results")

EXPERIMENTS = [
    {
        "row_label":  r"JARVIS $E_{\rm g}$",
        "theta":      r"$\theta = 0.4$",
        "threshold":  0.4,
        "dir":        os.path.join(BASE, "jarvis_eg_learnable_cutoff_scoring_net_th0_4_sw0_001_0506"),
        "json_key":   "pbe_eg_bvm_struct",
        "png_name":   "explainability_test_jarvis_lg_cell_eg_0.png",
        "group":      "JARVIS",
    },
    {
        "row_label":  r"JARVIS $E_{\rm f}$",
        "theta":      r"$\theta = 0.3$",
        "threshold":  0.3,
        "dir":        os.path.join(BASE, "jarvis_ef_learnable_cutoff_scoring_net_th0_3_sw0_001_0510"),
        "json_key":   "jarvis_lg_gap",
        "png_name":   "explainability_test_jarvis_lg_cell_ef_0.png",
        "group":      "JARVIS",
    },
    {
        "row_label":  r"MP $E_{\rm g}$",
        "theta":      r"$\theta = 0.2$",
        "threshold":  0.2,
        "dir":        os.path.join(BASE, "mp_eg_learnable_cutoff_scoring_net_th0_2_sw0_001_0506"),
        "json_key":   "lg_unit_cell_eg",
        "png_name":   "explainability_test_mp_lg_cell_eg_0.png",
        "group":      "MP",
    },
    {
        "row_label":  r"MP $E_{\rm f}$",
        "theta":      r"$\theta = 0.3$",
        "threshold":  0.3,
        "dir":        os.path.join(BASE, "mp_ef_learnable_cutoff_scoring_net_th0_3_sw0_001_0510"),
        "json_key":   "lg_unit_cell_ef",
        "png_name":   "explainability_test_mp_lg_cell_ef_0.png",
        "group":      "MP",
    },
]


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


def regenerate_png_from_json(exp: dict) -> str:
    """Regenerate the explainability PNG from the prob_dict JSON with manuscript fonts.

    Tries several common filename patterns for the JSON.
    Returns the path of the written PNG, or None if no JSON was found.
    """
    key       = exp["json_key"]
    out_dir   = exp["dir"]
    threshold = exp["threshold"]

    # Filename pattern used by runner.py: test_<key>_<fold>_prob_dict.json
    candidates = [
        os.path.join(out_dir, f"test_{key}_0_prob_dict.json"),
        os.path.join(out_dir, f"{key}_test_0_prob_dict.json"),
        os.path.join(out_dir, f"{key}_0_prob_dict.json"),
    ]
    json_path = next((p for p in candidates if os.path.exists(p)), None)
    if json_path is None:
        return None

    print(f"  Regenerating PNG from {os.path.basename(json_path)} ...")
    with open(json_path) as f:
        prob_data = json.load(f)

    all_weights, all_lengths = [], []
    for jid, data in prob_data.items():
        weights = data.get("prop_dist", [])
        lengths = data.get("bond_lengths", [])
        if len(weights) == len(lengths) and len(weights) > 0:
            all_weights.extend(weights)
            all_lengths.extend(lengths)

    if not all_weights:
        print(f"  Warning: no bond-length data found in {json_path}")
        return None

    all_weights = np.array(all_weights, dtype=float)
    all_lengths = np.array(all_lengths, dtype=float)
    kept_mask       = all_weights >= threshold
    kept_lengths    = all_lengths[kept_mask]
    dropped_lengths = all_lengths[~kept_mask]

    # Okabe-Ito colorblind-safe palette
    C_KEPT    = "#0072B2"   # deep blue  — retained edges
    C_DROPPED = "#E69F00"   # warm amber — pruned edges
    C_THRESH  = "#CC3311"   # vivid red  — threshold line

    with matplotlib.rc_context(MANUSCRIPT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter: edge weight vs bond length
        axes[0].scatter(all_lengths, all_weights, alpha=0.18, s=2, c=C_KEPT,
                        rasterized=True)
        axes[0].axhline(y=threshold, color=C_THRESH, linestyle="--",
                        linewidth=2.0, label=f"threshold={threshold}")
        axes[0].set_xlabel("Bond Length (\u00c5)")
        axes[0].set_ylabel("Edge Weight")
        axes[0].legend()

        # Histogram: kept vs dropped
        axes[1].hist(kept_lengths, bins=50, alpha=0.65, density=True,
                     label=f"Kept ({kept_mask.sum()})", color=C_KEPT,
                     edgecolor="white", linewidth=0.3)
        axes[1].hist(dropped_lengths, bins=50, alpha=0.65, density=True,
                     label=f"Dropped ({(~kept_mask).sum()})", color=C_DROPPED,
                     edgecolor="white", linewidth=0.3)
        axes[1].set_xlabel("Bond Length (\u00c5)")
        axes[1].set_ylabel("Density")
        axes[1].legend()

        plt.tight_layout()
        png_path = os.path.join(out_dir, exp["png_name"])
        plt.savefig(png_path, bbox_inches="tight", dpi=200)
        plt.close()

    return png_path

# ---------------------------------------------------------------------------
# Regenerate source PNGs from JSON if available, else use existing PNGs
# ---------------------------------------------------------------------------
print("Checking for JSON data files ...")
for exp in EXPERIMENTS:
    png_path = os.path.join(exp["dir"], exp["png_name"])
    result = regenerate_png_from_json(exp)  # JSON takes priority over existing PNG
    if result:
        print(f"  Regenerated: {os.path.basename(result)}")
    elif os.path.exists(png_path):
        print(f"  Using existing PNG: {exp['png_name']}")
    else:
        raise FileNotFoundError(
            f"Neither JSON nor PNG found for {exp['row_label']}.\n"
            f"  Expected PNG:  {png_path}\n"
            f"  Expected JSON: {os.path.join(exp['dir'], 'test_' + exp['json_key'] + '_0_prob_dict.json')}"
        )

# ---------------------------------------------------------------------------
# Matplotlib / font settings for manuscript quality
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        11,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})

PANEL_LETTERS = list("abcdefgh")


def load_and_split(path: str):
    """Load PNG, remove internal titles + white borders, split into left/right halves."""
    print(f"Loading and processing {os.path.basename(path)} ...")
    img = np.array(Image.open(path).convert("RGB"))
    H, W = img.shape[:2]

    # --- remove internal matplotlib title (rows 0..~40) and bottom whitespace ---
    # The title text occupies roughly rows 15-33; rows 34-40 are pure white.
    # Start from row 40 to remove the title cleanly while keeping axis labels.
    TITLE_CROP_TOP = 40
    row_means = img.mean(axis=(1, 2))
    non_white_rows = np.where(row_means < 248)[0]
    r0 = TITLE_CROP_TOP
    r1 = min(non_white_rows[-1] + 4, H)
    img = img[r0:r1, :, :]

    # --- split at midpoint (white gutter between the two sub-plots) ---
    mid = W // 2
    left  = img[:, :mid,  :]
    right = img[:, mid:,  :]

    # --- trim horizontal white borders from each half ---
    def trim_h(half):
        col_means = half.mean(axis=(0, 2))
        cols = np.where(col_means < 248)[0]
        if len(cols) == 0:
            return half
        return half[:, max(cols[0] - 2, 0):min(cols[-1] + 2, half.shape[1]), :]

    return trim_h(left), trim_h(right)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
N_ROWS = len(EXPERIMENTS)   # 4
N_COLS = 2                  # scatter | histogram

# Figure dimensions: 7.0 in wide (fits a 2-column journal page), height scaled
FIG_W = 14.0   # will be scaled to ~7 in in the PDF via \includegraphics
FIG_H = 10.5   # rows × aspect

fig = plt.figure(figsize=(FIG_W, FIG_H))

# Outer gridspec: 4 rows, 1 col — used to insert a horizontal rule between
# the JARVIS block (rows 0-1) and the MP block (rows 2-3).
outer = gridspec.GridSpec(
    N_ROWS, 1,
    figure=fig,
    hspace=0.10,      # tight row spacing
    top=0.95,
    bottom=0.03,
    left=0.10,
    right=0.93,       # leave right margin for group labels
)

axes = []
for row_idx, exp in enumerate(EXPERIMENTS):
    # Each outer row gets a 1×2 inner gridspec
    inner = gridspec.GridSpecFromSubplotSpec(
        1, N_COLS,
        subplot_spec=outer[row_idx],
        wspace=0.06,
    )
    row_axes = []
    for col_idx in range(N_COLS):
        ax = fig.add_subplot(inner[col_idx])
        row_axes.append(ax)
    axes.append(row_axes)

# ---------------------------------------------------------------------------
# Fill each sub-plot with its cropped image
# ---------------------------------------------------------------------------
for row_idx, exp in enumerate(EXPERIMENTS):
    left_img, right_img = load_and_split(os.path.join(exp["dir"], exp["png_name"]))
    imgs = [left_img, right_img]
    col_titles = [
        "Edge Weight vs. Bond Length",
        "Bond Length: Kept vs. Dropped",
    ]
    for col_idx, (ax, im, ctitle) in enumerate(zip(axes[row_idx], imgs, col_titles)):
        ax.imshow(im, interpolation="lanczos", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Column headers (top row only)
        if row_idx == 0:
            ax.set_title(ctitle, fontsize=13, fontweight="bold", pad=8)

        # Panel letter label  (a)–(h) in top-left corner
        letter = PANEL_LETTERS[row_idx * N_COLS + col_idx]
        ax.text(
            0.015, 0.962, f"({letter})",
            transform=ax.transAxes,
            fontsize=13, fontweight="bold",
            va="top", ha="left",
            color="black",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.80),
        )

    # Row label on the left (dataset + θ)
    ax0 = axes[row_idx][0]
    ax0.text(
        -0.015, 0.5,
        exp["row_label"] + f"\n{exp['theta']}",
        transform=ax0.transAxes,
        fontsize=12, fontweight="bold",
        va="center", ha="right",
        rotation=0,
        multialignment="center",
    )

# ---------------------------------------------------------------------------
# Horizontal separator line between JARVIS and MP blocks
# ---------------------------------------------------------------------------
# Get figure coordinates of the bottom of row 1 and top of row 2
ax_bottom = axes[1][0]  # last JARVIS row, left subplot
ax_top    = axes[2][0]  # first MP row, left subplot

bb1 = ax_bottom.get_position()
bb2 = ax_top.get_position()
y_sep = (bb1.y0 + bb2.y1) / 2.0

line = Line2D(
    [0.02, 0.97], [y_sep, y_sep],
    transform=fig.transFigure,
    color="#888888",
    linewidth=1.2,
    linestyle="--",
    clip_on=False,
)
fig.add_artist(line)

# ---------------------------------------------------------------------------
# Group labels in right margin (rotated vertical text)
# ---------------------------------------------------------------------------
# "JARVIS" centered between rows 0–1
bb_j0 = axes[0][1].get_position()
bb_j1 = axes[1][1].get_position()
y_jarvis = (bb_j0.y1 + bb_j1.y0) / 2.0

# "Materials Project" centered between rows 2–3
bb_m0 = axes[2][1].get_position()
bb_m1 = axes[3][1].get_position()
y_mp = (bb_m0.y1 + bb_m1.y0) / 2.0

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
# Save
# ---------------------------------------------------------------------------
out_path = os.path.join(os.path.dirname(__file__), "explainability_figure.pdf")
png_path = os.path.join(os.path.dirname(__file__), "explainability_figure.png")

fig.savefig(out_path, dpi=300, format="pdf")
fig.savefig(png_path, dpi=300, format="png")
print(f"Saved:\n  {out_path}\n  {png_path}")
