import json
import glob
import numpy as np

# ---------------------------------------------------------
# 1. Configuration: Update these two variables!
# ---------------------------------------------------------
# Use a wildcard to grab all 5 JSON files from the CV run
json_pattern = "./gen_phase1_jarvis_Eg_learnable_cutoff_scoring_net_th0_4_sw0_001_0608/*_prob_dict.json" 

# Set the threshold used for this specific experiment 
# (e.g., 0.4 for JARVIS Eg, 0.2 for MP Eg)
threshold = 0.4
# ---------------------------------------------------------

json_files = sorted(glob.glob(json_pattern))

if not json_files:
    print(f"No files found matching pattern: {json_pattern}")
    exit()

print(f"Found {len(json_files)} files. Calculating sparsity for threshold = {threshold}...\n")

fold_percentages = []
fold_kept_counts = []

# 2. Process each fold
for file_path in json_files:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    total_edges_in_fold = 0
    kept_edges_in_fold = 0
    
    # Process every crystal structure in this specific fold
    for jid, crystal_data in data.items():
        weights = crystal_data.get("prop_dist", [])
        
        if not weights:
            continue
            
        weights_arr = np.array(weights)
        
        # Tally the edges
        total_edges_in_fold += len(weights_arr)
        kept_edges_in_fold += np.sum(weights_arr >= threshold)
        
    # Calculate stats for this specific fold
    if total_edges_in_fold > 0:
        fold_pct = (kept_edges_in_fold / total_edges_in_fold) * 100
        fold_percentages.append(fold_pct)
        fold_kept_counts.append(kept_edges_in_fold)
        
        filename = file_path.split('/')[-1]
        print(f"Fold: {filename}")
        print(f"  -> Kept: {kept_edges_in_fold:,} / {total_edges_in_fold:,} edges ({fold_pct:.2f}%)")

# 3. Final Aggregation
if fold_percentages:
    mean_pct = np.mean(fold_percentages)
    std_pct = np.std(fold_percentages)
    avg_kept_count = np.mean(fold_kept_counts)
    
    print("\n" + "="*50)
    print("FINAL 5-FOLD CV RESULTS:")
    print("="*50)
    print(f"Average Kept Percentage:    {mean_pct:.2f}% ± {std_pct:.2f}%")
    print(f"Average Dropped Percentage: {100 - mean_pct:.2f}%")
    print(f"Average Kept Edges per run: {avg_kept_count:,.0f}")
