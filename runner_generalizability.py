"""runner_generalizability.py

Runner extension for the cross-property generalizability experiment.

Inherits all training/evaluation/prediction logic from the base Runner and
adds three utilities:

    collect_test_broadcast_graphs(phase1_output_dir, n_folds, consolidated_dir)
        Pool per-fold test-split broadcast graphs into one flat directory.
        In 5-fold CV each material appears in exactly one test fold, so this
        gives *unbiased* Phase-1 broadcast graphs covering all structures.

    ensure_graph_dirs(output_dir, n_folds, splits)
        Pre-create the {split}_{fold}_graphs/ subdirectories that
        runner.predict() will try to write into.  DGL's save_graphs does not
        create missing parent directories, so this must be called before
        runner.test().

    freeze_action_net()
        Freeze all action-network parameters in-place (useful for a
        frozen-net transfer-learning variant of the experiment).
"""

import glob
import os
import shutil

from runner import Runner  # base class – all standard training logic lives here


class GeneralizabilityRunner(Runner):
    """Runner extended with graph-collection and net-freezing utilities."""

    # ── Static / class helpers ─────────────────────────────────────────────────

    @staticmethod
    def collect_test_broadcast_graphs(
        phase1_output_dir: str,
        n_folds: int,
        consolidated_dir: str,
    ) -> int:
        """Copy {jid}_bcast.bin from every test_{fold}_graphs/ into one dir.

        In a clean 5-fold CV each material appears in exactly one test fold,
        so the union of test_0_graphs … test_4_graphs covers all structures
        exactly once with unbiased (held-out) broadcast graphs.

        Args:
            phase1_output_dir : base output directory of the Phase-1 run,
                                 e.g. ``outputs/gen_phase1_Ef_lcsn_th0_3_sw0_001_0512``.
            n_folds           : number of CV folds used in Phase 1.
            consolidated_dir  : destination directory (created if absent).
                                 **Must end with '/'** because graphs.py uses
                                 it as a path prefix: ``trim_graph_dir + jid + '_bcast.bin'``.

        Returns:
            Number of files copied.
        """
        if not consolidated_dir.endswith("/"):
            consolidated_dir = consolidated_dir + "/"
        os.makedirs(consolidated_dir, exist_ok=True)

        total = 0
        for fold in range(n_folds):
            src = os.path.join(phase1_output_dir, f"test_{fold}_graphs")
            if not os.path.isdir(src):
                print(f"[collect] WARNING: {src} not found — fold {fold} skipped")
                continue
            files = glob.glob(os.path.join(src, "*_bcast.bin"))
            for fpath in files:
                dst = os.path.join(consolidated_dir, os.path.basename(fpath))
                if not os.path.exists(dst):
                    shutil.copy2(fpath, dst)
                    total += 1
        print(f"[collect] {total} broadcast graphs → {consolidated_dir}")
        return total

    @staticmethod
    def ensure_graph_dirs(
        output_dir: str,
        n_folds: int,
        splits: tuple = ("train", "val", "test"),
    ) -> None:
        """Pre-create per-fold graph directories so dgl.save_graphs won't fail.

        runner.predict() writes to paths like::

            ./outputs/{output_dir}/test_0_graphs/{jid}_bcast.bin

        but does not create the ``test_0_graphs/`` sub-directory itself,
        which causes a FileNotFoundError.  Call this before runner.test().
        """
        for fold in range(n_folds):
            for split in splits:
                path = os.path.join(".", output_dir, f"{split}_{fold}_graphs")
                os.makedirs(path, exist_ok=True)

    # ── Instance helpers ───────────────────────────────────────────────────────

    def freeze_action_net(self) -> list:
        """Freeze all action-network parameters in-place.

        Covers the four action-net variants (EdgeScoringNet,
        EdgeOnlyScoringNet, LearnableCutoffScoringNet,
        MultiplicativeGateScoringNet) as well as the legacy ActionNet
        parameter names.

        Returns:
            List of frozen parameter names.
        """
        action_name_fragments = (
            "edge_scoring_net",
            "in_act_graph_net",
            "out_act_graph_net",
            "in_act_linegraph_net",
            "out_act_linegraph_net",
        )
        frozen = []
        for name, param in self.model.named_parameters():
            if any(frag in name for frag in action_name_fragments):
                param.requires_grad_(False)
                frozen.append(name)

        n_frozen    = len(frozen)
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[freeze_action_net] {n_frozen} parameter tensors frozen | "
              f"trainable params remaining: {n_trainable:,}")
        return frozen
