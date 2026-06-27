"""
Round 2 layout fix — Level-2 DFD 7.5 Cox Survival Analysis Tuning.

Fixes from round1:
- Process boxes widened so "Fit CoxnetSurvivalAnalysis" and "Harrell concordance" no longer clip
- Horizontal spacing increased between all 6 sub-processes
- Internal edge labels raised 12 px above edge lines (label_y_offset=-12)
- Output edges to D4 use separate exit styles (exitY=0.3/0.7) to prevent label merging
- Canvas enlarged to 1400×900
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))
from shared.base_sub_dfd_builder import SubDFDBuilder

DIAGRAMS_DIR = Path(__file__).parents[4] / "diagrams"


class CoxTuningSubDFDv2(SubDFDBuilder):
    OUTPUT_FILENAME = "Level-2 DFD - 7.5 cox survival analysis tuning.drawio"
    PARENT_LABEL   = "7.5 Cox Survival Analysis Tuning"
    CONTAINER_X    = 210
    CONTAINER_W    = 950
    CONTAINER_H    = 230
    PAGE_WIDTH     = 1400
    PAGE_HEIGHT    = 900

    def build(self):
        # Row 1: 7.5.1 → 7.5.2 → 7.5.3 → 7.5.4
        # Widened 7.5.3 (+40 px) and 7.5.4 (+45 px); all boxes taller (+10 px)
        p1_o, p1_i = self.add_process(
            "7.5.1", "Initialize<div>hyperparameter grid</div><div>6 alpha x 3 l1_ratio</div>",
            20, 40, w=160, h=80,
        )
        p2_o, p2_i = self.add_process(
            "7.5.2", "K-Fold<div>cross-validation</div>",
            205, 40, w=145, h=80,
        )
        p3_o, p3_i = self.add_process(
            "7.5.3", "Fit CoxnetSurvivalAnalysis<div>per fold</div>",
            375, 40, w=200, h=80,
        )
        p4_o, p4_i = self.add_process(
            "7.5.4", "Score C-index<div>(Harrell concordance)</div>",
            600, 40, w=185, h=80,
        )

        # Row 2: 7.5.5 → 7.5.6
        p5_o, p5_i = self.add_process(
            "7.5.5", "Select best<div>(alpha, l1_ratio)</div>",
            205, 160, w=155, h=60,
        )
        p6_o, p6_i = self.add_process(
            "7.5.6", "Derive 9 optimal<div>time points</div>",
            385, 160, w=155, h=60,
        )

        # Internal edges — labels raised above the edge line
        self.add_edge(p1_i, p2_i, "Grid combos (18)", label_y_offset=-12)
        self.add_edge(p2_i, p3_i, "Fold splits",      label_y_offset=-12)
        self.add_edge(p3_i, p4_i, "Fitted model",     label_y_offset=-12)
        self.add_edge(p4_i, p5_i, "C-index scores",   label_y_offset=-12)
        self.add_edge(p5_i, p6_i, "Best params",      label_y_offset=-12)

        # Input: df_data_surv from D3
        self.add_edge(
            self.store_id("D3"), p1_o,
            "df_data_surv<div>(survival stream)</div>",
            parent_id="1",
        )

        # Outputs to D4 — staggered exit points so labels don't merge
        self.add_edge_styled(
            p5_o, self.store_id("D4"),
            "Best (alpha, l1_ratio)",
            parent_id="1",
            style_extra="exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=1;entryY=0.3;entryDx=0;entryDy=0;",
            label_y_offset=-12,
        )
        self.add_edge_styled(
            p6_o, self.store_id("D4"),
            "9 optimal time points",
            parent_id="1",
            style_extra="exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=1;entryY=0.7;entryDx=0;entryDy=0;",
            label_y_offset=12,
        )

        print("[7.5 v2] Built Cox Survival Analysis Tuning sub-DFD: 6 processes (layout fixed)")


if __name__ == "__main__":
    builder = CoxTuningSubDFDv2(DIAGRAMS_DIR)
    builder.build()
    builder.save()
