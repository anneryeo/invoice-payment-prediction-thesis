"""Level-2 DFD — 7.5 Cox Survival Analysis Tuning sub-processes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))
from shared.base_sub_dfd_builder import SubDFDBuilder

DIAGRAMS_DIR = Path(__file__).parents[4] / "diagrams"


class CoxTuningSubDFD(SubDFDBuilder):
    OUTPUT_FILENAME = "Level-2 DFD - 7.5 cox survival analysis tuning.drawio"
    PARENT_LABEL   = "7.5 Cox Survival Analysis Tuning"
    CONTAINER_W    = 760
    CONTAINER_H    = 200

    def build(self):
        # Row 1: 7.5.1 -> 7.5.2 -> 7.5.3
        p1_o, p1_i = self.add_process("7.5.1", "Initialize<div>hyperparameter grid</div><div>6 alpha x 3 l1_ratio</div>",
                                       20, 40, w=145, h=70)
        p2_o, p2_i = self.add_process("7.5.2", "K-Fold<div>cross-validation</div>",
                                       190, 40, w=120, h=70)
        p3_o, p3_i = self.add_process("7.5.3", "Fit CoxnetSurvivalAnalysis<div>per fold</div>",
                                       335, 40, w=155, h=70)
        p4_o, p4_i = self.add_process("7.5.4", "Score C-index<div>(Harrell concordance)</div>",
                                       515, 40, w=130, h=70)

        # Row 2: 7.5.5 -> 7.5.6
        p5_o, p5_i = self.add_process("7.5.5", "Select best<div>(alpha, l1_ratio)</div>",
                                       200, 140, w=130, h=55)
        p6_o, p6_i = self.add_process("7.5.6", "Derive 9 optimal<div>time points</div>",
                                       370, 140, w=130, h=55)

        self.add_edge(p1_i, p2_i, "Grid combos (18)")
        self.add_edge(p2_i, p3_i, "Fold splits")
        self.add_edge(p3_i, p4_i, "Fitted model")
        self.add_edge(p4_i, p5_i, "C-index scores")
        self.add_edge(p5_i, p6_i, "Best params")

        # Input: df_data_surv from D3
        self.add_edge(self.store_id("D3"), p1_o, "df_data_surv<div>(survival stream)</div>", parent_id="1")

        # Outputs to D4 (Models store) — best parameters and time points
        self.add_edge(p5_o, self.store_id("D4"), "Best (alpha, l1_ratio)", parent_id="1")
        self.add_edge(p6_o, self.store_id("D4"), "9 optimal time points", parent_id="1")

        print("[7.5] Built Cox Survival Analysis Tuning sub-DFD: 6 processes")


if __name__ == "__main__":
    builder = CoxTuningSubDFD(DIAGRAMS_DIR)
    builder.build()
    builder.save()
