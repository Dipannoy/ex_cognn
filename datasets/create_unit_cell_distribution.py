import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# ---------------------------------------------------------------------------
# 1. Dummy Data Loader (Replace with your actual data loading)
# ---------------------------------------------------------------------------
def get_atom_counts(pickle_path):
    """
    Extracts the number of atoms from your pre-processed pickle dataset.
    Adjust the keys based on your exact pickle structure.
    """
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        # Assuming data is a list of dictionaries where 'atoms' contains 'elements'
        # based on your data_prep.py: natoms = len(d['elements'])
        counts = [len(item['atoms']['elements']) for item in data]
        return np.array(counts)
    except Exception as e:
        print(f"Could not load {pickle_path}. Generating dummy data for demonstration.")
        # Fallback dummy data just to show the plot works
        return np.random.lognormal(mean=3.0, sigma=0.8, size=10000).astype(int)

# Load your actual pickle files here
jarvis_counts = get_atom_counts('./jarvis/Jarvis_full_ef/jarvis_cell_ef_full.pickle')
mp_counts = get_atom_counts('./mp/MP_full_ef/mp_l_cell_ef_full.pickle')

# ---------------------------------------------------------------------------
# 2. Plot Configuration
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titleweight": "bold",
    "legend.fontsize": 12,
    "figure.dpi": 300
})

C_JARVIS = "#009E73"  # Teal
C_MP = "#CC79A7"      # Magenta/Pink

fig, ax = plt.subplots(figsize=(10, 6))

# ---------------------------------------------------------------------------
# 3. Plotting Distributions
# ---------------------------------------------------------------------------
# Use seaborn to plot overlapping histograms with KDE (Kernel Density Estimate)
sns.histplot(jarvis_counts, bins=75, color=C_JARVIS, label='JARVIS', 
             kde=True, stat='density', alpha=0.5, binrange=(0, 300), ax=ax)
sns.histplot(mp_counts, bins=75, color=C_MP, label='Materials Project', 
             kde=True, stat='density', alpha=0.5, binrange=(0, 300), ax=ax)

# Add vertical lines for the medians
jarvis_median = np.median(jarvis_counts)
mp_median = np.median(mp_counts)

ax.axvline(jarvis_median, color=C_JARVIS, linestyle='dashed', linewidth=2, 
           label=f'JARVIS Median: {jarvis_median:.0f}')
ax.axvline(mp_median, color=C_MP, linestyle='dashed', linewidth=2, 
           label=f'MP Median: {mp_median:.0f}')

# ---------------------------------------------------------------------------
# 4. Formatting
# ---------------------------------------------------------------------------
# ax.set_title("Unit Cell Size Distribution Across Datasets")
ax.set_xlabel("Number of Atoms per Unit Cell")
ax.set_ylabel("Density")
ax.set_xlim(0, 300) # Cut off extreme outliers for a cleaner plot

# Grid and Legend
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
ax.legend(loc='upper right', frameon=True, shadow=True)

plt.tight_layout()
plt.savefig('unit_cell_distribution_full_dataset.pdf', format='pdf', bbox_inches='tight')
plt.savefig('unit_cell_distribution_full_dataset.png', format='png', bbox_inches='tight')
print("Plot saved to unit_cell_distribution_full_dataset.pdf and unit_cell_distribution_full_dataset.png")
