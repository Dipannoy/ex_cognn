"""data_prep_generalizability.py

Prepares and verifies the MP20 Ef and Eg datasets for the cross-property
generalizability experiment.

Steps executed
--------------
1. Load both pickles and report sizes.
2. Confirm JID overlap and ordering (required for consistent 5-fold CV splits).
3. If either dataset is larger, create intersection-filtered pickles so that
   Phase 1 and Phase 2 work on exactly the same set of structures.
4. Create datasets/mp/ef/config_ef_coalignn.json (proper Ef-specific config).
5. Print a plain-text summary so it can be copy-pasted into the paper.

Usage
-----
    cd ex_cognn
    python data_prep_generalizability.py [--check_only]
"""

import argparse
import json
import os
import pickle

import pandas as pd
from omegaconf import OmegaConf


# ── Default paths ──────────────────────────────────────────────────────────────
EF_PKL    = "datasets/mp/ef/mp_lg_unit_cell_ef.pickle"
EG_PKL    = "datasets/mp/mp_lg_unit_cell_gap.pickle"
EF_CSV    = "datasets/mp/ef/mp_ef_lg_unit_cell_id.csv"
EG_CSV    = "datasets/mp/mp_gap_lg_unit_cell_id.csv"
EG_CONFIG = "datasets/mp/config_eg_coalignn.json"
EF_CONFIG = "datasets/mp/ef/config_ef_coalignn.json"

# Output prefix when filtered pickles are needed
EF_PKL_GEN = "datasets/mp/ef/mp_lg_unit_cell_ef_gen.pickle"
EG_PKL_GEN = "datasets/mp/mp_lg_unit_cell_gap_gen.pickle"
EF_CSV_GEN = "datasets/mp/ef/mp_ef_lg_unit_cell_id_gen.csv"
EG_CSV_GEN = "datasets/mp/mp_gap_lg_unit_cell_id_gen.csv"


def load_pickle(path: str) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


def check_target_key(dataset: list, key: str) -> bool:
    """Return True if all records have the expected target key."""
    missing = sum(1 for d in dataset if key not in d)
    if missing > 0:
        print(f"  WARNING: {missing}/{len(dataset)} records missing key '{key}'")
        # Try to find the actual key
        sample = dataset[0]
        candidates = [k for k in sample if k not in ("jid", "atoms", "system", "nf")]
        print(f"  Available keys in first record: {candidates}")
        return False
    return True


def create_ef_config(eg_config_path: str, ef_config_path: str):
    """Derive the Ef training config from the existing Eg config."""
    cfg = OmegaConf.load(eg_config_path)

    cfg.target     = "Ef"
    cfg.suffix     = "lg_unit_cell_ef"
    cfg.output_dir = "output_config_ef_coalignn"

    OmegaConf.save(cfg, ef_config_path)
    print(f"\nCreated Ef config → {ef_config_path}")
    print(f'  target = "Ef", suffix = "lg_unit_cell_ef"')


def main(check_only: bool = False):
    print("=" * 60)
    print("MP20 Generalizability Data Preparation")
    print("=" * 60)

    # ── 1. Load both pickles ───────────────────────────────────────────────────
    print(f"\n[1] Loading pickles …")
    ef_data = load_pickle(EF_PKL)
    eg_data = load_pickle(EG_PKL)

    ef_jid_list = [d["jid"] for d in ef_data]
    eg_jid_list = [d["jid"] for d in eg_data]
    ef_jids = set(ef_jid_list)
    eg_jids = set(eg_jid_list)

    print(f"  Ef: {len(ef_data):,} structures | {len(ef_jids):,} unique JIDs")
    print(f"  Eg: {len(eg_data):,} structures | {len(eg_jids):,} unique JIDs")

    # ── 2. Check target keys ───────────────────────────────────────────────────
    print(f"\n[2] Checking target keys …")
    ef_ok = check_target_key(ef_data, "Ef")
    eg_ok = check_target_key(eg_data, "Eg")
    print(f"  Ef key 'Ef' present: {ef_ok}")
    print(f"  Eg key 'Eg' present: {eg_ok}")

    # ── 3. JID overlap analysis ────────────────────────────────────────────────
    print(f"\n[3] JID overlap …")
    shared   = ef_jids & eg_jids
    ef_only  = ef_jids - eg_jids
    eg_only  = eg_jids - ef_jids

    pct = 100.0 * len(shared) / max(len(ef_jids), len(eg_jids))
    print(f"  Shared JIDs : {len(shared):,}  ({pct:.2f}%)")
    print(f"  Ef-only JIDs: {len(ef_only):,}")
    print(f"  Eg-only JIDs: {len(eg_only):,}")

    order_match = ef_jid_list[:len(shared)] == eg_jid_list[:len(shared)]
    print(f"  JID order identical (full list): {ef_jid_list == eg_jid_list}")

    # ── 4. Create filtered pickles if necessary ────────────────────────────────
    needs_filter = (len(ef_only) > 0 or len(eg_only) > 0 or
                    ef_jid_list != eg_jid_list)

    if needs_filter and not check_only:
        print(f"\n[4] Creating intersection-filtered pickles …")

        # Use Eg order as canonical (it is typically the reference set)
        canonical_order = [jid for jid in eg_jid_list if jid in ef_jids]
        ef_idx  = {d["jid"]: d for d in ef_data}
        eg_idx  = {d["jid"]: d for d in eg_data}

        ef_filtered = [ef_idx[jid] for jid in canonical_order]
        eg_filtered = [eg_idx[jid] for jid in canonical_order]

        with open(EF_PKL_GEN, "wb") as f:
            pickle.dump(ef_filtered, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(EG_PKL_GEN, "wb") as f:
            pickle.dump(eg_filtered, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Also write aligned CSVs from the originals
        ef_csv = pd.read_csv(EF_CSV)
        eg_csv = pd.read_csv(EG_CSV)
        ef_csv["jid"] = ef_csv["filename"].str.split(".").str[0]
        eg_csv["jid"] = eg_csv["filename"].str.split(".").str[0]
        ef_csv_f = ef_csv[ef_csv["jid"].isin(set(canonical_order))].set_index("jid").loc[canonical_order].reset_index()
        eg_csv_f = eg_csv[eg_csv["jid"].isin(set(canonical_order))].set_index("jid").loc[canonical_order].reset_index()
        ef_csv_f.to_csv(EF_CSV_GEN, index=False)
        eg_csv_f.to_csv(EG_CSV_GEN, index=False)

        print(f"  Filtered Ef pickle  → {EF_PKL_GEN}  ({len(ef_filtered):,} records)")
        print(f"  Filtered Eg pickle  → {EG_PKL_GEN}  ({len(eg_filtered):,} records)")
        print(f"  Filtered Ef CSV     → {EF_CSV_GEN}")
        print(f"  Filtered Eg CSV     → {EG_CSV_GEN}")
        print(f"\n  *** Use _gen pickles + CSVs in run_mp20_gen.sh ***")

    elif not needs_filter:
        print(f"\n[4] Datasets are already perfectly aligned — originals can be used as-is.")
        EF_PKL_GEN_USE = EF_PKL
        EG_PKL_GEN_USE = EG_PKL
        EF_CSV_GEN_USE = EF_CSV
        EG_CSV_GEN_USE = EG_CSV
    else:
        print(f"\n[4] check_only mode — skipping filtered pickle creation.")

    # ── 5. Create Ef config ────────────────────────────────────────────────────
    if not check_only:
        if os.path.exists(EF_CONFIG):
            # Verify it has the right target
            existing = OmegaConf.load(EF_CONFIG)
            if existing.get("target") != "Ef":
                print(f"\n[5] Overwriting mismatched Ef config ({EF_CONFIG}) …")
                create_ef_config(EG_CONFIG, EF_CONFIG)
            else:
                print(f"\n[5] Ef config already correct: {EF_CONFIG}")
        else:
            create_ef_config(EG_CONFIG, EF_CONFIG)

    # ── 6. Summary for paper ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY (for paper / note-keeping)")
    print("=" * 60)
    print(f"  Dataset       : MP20")
    print(f"  Ef structures : {len(ef_data):,}")
    print(f"  Eg structures : {len(eg_data):,}")
    print(f"  Shared (∩)    : {len(shared):,} ({pct:.1f}%)")
    print(f"  Ef target key : 'Ef'  (formation energy per atom, eV)")
    print(f"  Eg target key : 'Eg'  (DFT band gap, eV)")
    print()
    print("  Generalizability experiment design:")
    print("    Phase 1 → Train LearnableCutoffScoringNet on Ef")
    print("              → generates pruned graphs for all structures")
    print("    Phase 2 → Train CoALIGNN (no action net) on Eg")
    print("              → uses Phase 1 pruned graphs (trim_graph_dir)")
    print("    Claim   → If MAE_Eg_phase2 ≈ MAE_Eg_baseline,")
    print("              the Ef-learned pruning generalises across properties")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MP20 generalizability data prep")
    parser.add_argument("--check_only", action="store_true",
                        help="Only report statistics, do not write any files.")
    args = parser.parse_args()
    main(check_only=args.check_only)
