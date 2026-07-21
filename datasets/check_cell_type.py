import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ---------------------------------------------------------------------------
# 1. Data Loader with PyMatGen Reconstruction
# ---------------------------------------------------------------------------

def print_cell_statistics(orig_counts, prim_counts, conv_counts, dataset_name):
    total = len(orig_counts)
    if total == 0:
        print(f"No data available for {dataset_name}.")
        return

    # Logical masking to categorize each structure
    is_primitive = (orig_counts == prim_counts)
    is_conventional = (orig_counts == conv_counts) & (orig_counts != prim_counts)
    is_supercell = (orig_counts > conv_counts)
    
    # Calculate sums
    num_prim = np.sum(is_primitive)
    num_conv = np.sum(is_conventional)
    num_super = np.sum(is_supercell)
    
    # Catch edge cases (e.g., failed symmetry analysis where orig is somehow less than conv)
    num_other = total - (num_prim + num_conv + num_super)

    print(f"--- {dataset_name} Cell Type Summary ---")
    print(f"Total Structures:   {total}")
    print(f"Primitive Cells:    {num_prim} ({(num_prim/total)*100:.2f}%)")
    print(f"Conventional Cells: {num_conv} ({(num_conv/total)*100:.2f}%)")
    print(f"Supercells:         {num_super} ({(num_super/total)*100:.2f}%)")
    if num_other > 0:
        print(f"Other/Errors:       {num_other} ({(num_other/total)*100:.2f}%)")
    print("-" * 40 + "\n")

def get_cell_distributions(pickle_path, symprec=0.01):
    """
    Extracts the original, primitive, and conventional atom counts 
    by reconstructing the PyMatGen Structure from the pickle data.
    """
    orig_counts = []
    prim_counts = []
    conv_counts = []
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            
        print(f"Processing {len(data)} structures from {pickle_path}...")
        
        for item in tqdm(data, desc="Analyzing Symmetry"):
            # Extract the stored properties based on data_prep.py
            atoms_dict = item['atoms']
            lattice_matrix = atoms_dict['lattice_mat']
            species = atoms_dict['elements']
            coords = atoms_dict['coords'] # Fractional coordinates ('abc')
            
            # Reconstruct the PyMatGen Structure
            struct = Structure(lattice_matrix, species, coords)
            orig_atoms = len(struct)
            
            # Analyze symmetry to get primitive and conventional structures
            try:
                analyzer = SpacegroupAnalyzer(struct, symprec=symprec)
                prim_struct = analyzer.get_primitive_standard_structure()
                conv_struct = analyzer.get_conventional_standard_structure()
                
                prim_atoms = len(prim_struct)
                conv_atoms = len(conv_struct)
            except Exception as e:
                # Fallback if symmetry analyzer fails for a specific highly-distorted structure
                prim_atoms = orig_atoms
                conv_atoms = orig_atoms
                
            orig_counts.append(orig_atoms)
            prim_counts.append(prim_atoms)
            conv_counts.append(conv_atoms)
            
        return np.array(orig_counts), np.array(prim_counts), np.array(conv_counts)
        
    except Exception as e:
        print(f"Error loading {pickle_path}: {e}")
        return np.array([]), np.array([]), np.array([])

# Load your actual pickle files here
print("Loading JARVIS dataset...")
jarvis_orig, jarvis_prim, jarvis_conv = get_cell_distributions('./jarvis/jarvis_lg_cell_gap.pickle')

print("Loading Materials Project dataset...")
mp_orig, mp_prim, mp_conv = get_cell_distributions('./mp/mp_lg_unit_cell_gap.pickle')

print_cell_statistics(jarvis_orig, jarvis_prim, jarvis_conv, "JARVIS")
print_cell_statistics(mp_orig, mp_prim, mp_conv, "Materials Project")

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

# ---------------------------------------------------------------------------
# 3. Plotting Primitive Cell Distributions
# ---------------------------------------------------------------------------
# You can change `jarvis_prim` and `mp_prim` to `jarvis_conv` or `jarvis_orig` 
# depending on exactly which distribution you want to visualize.

fig, ax = plt.subplots(figsize=(10, 6))

sns.histplot(jarvis_prim, bins=75, color=C_JARVIS, label='JARVIS (Primitive)', 
             kde=True, stat='density', alpha=0.5, binrange=(0, 300), ax=ax)
sns.histplot(mp_prim, bins=75, color=C_MP, label='Materials Project (Primitive)', 
             kde=True, stat='density', alpha=0.5, binrange=(0, 300), ax=ax)

# Add vertical lines for the medians
jarvis_median = np.median(jarvis_prim)
mp_median = np.median(mp_prim)

ax.axvline(jarvis_median, color=C_JARVIS, linestyle='dashed', linewidth=2, 
           label=f'JARVIS Median: {jarvis_median:.0f}')
ax.axvline(mp_median, color=C_MP, linestyle='dashed', linewidth=2, 
           label=f'MP Median: {mp_median:.0f}')

# ---------------------------------------------------------------------------
# 4. Formatting
# ---------------------------------------------------------------------------
ax.set_xlabel("Number of Atoms per Primitive Unit Cell")
ax.set_ylabel("Density")
ax.set_xlim(0, 300) # Cut off extreme outliers for a cleaner plot

# Grid and Legend
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
ax.legend(loc='upper right', frameon=True, shadow=True)

plt.tight_layout()
plt.savefig('primitive_cell_distribution.pdf', format='pdf', bbox_inches='tight')
plt.savefig('primitive_cell_distribution.png', format='png', bbox_inches='tight')
print("Plots successfully generated and saved!")