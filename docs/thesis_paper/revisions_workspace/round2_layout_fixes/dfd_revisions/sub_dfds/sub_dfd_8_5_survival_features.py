"""
Round 2 layout fix — Level-2 DFD 8.5 Survival Feature Generation.

Fixes from round1:
- Process boxes widened so "Fit CoxnetSurvivalAnalysis" and "E[T]" no longer clip.
- More horizontal spacing between the 4 row-1 processes.
- "Scalar features" edge (8.5.4 → 8.5.5) now exits from 8.5.4 bottom and enters
  8.5.5 from the right, routing around the 8.5.3 area instead of through it.
- S(t)/H(t) edge labels raised 12 px above edge lines (label_y_offset=-12).
- Canvas enlarged to 1400×900.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))
from shared.base_sub_dfd_builder import SubDFDBuilder

DIAGRAMS_DIR = Path(__file__).parents[4] / "diagrams"


class SurvivalFeatureSubDFDv2(SubDFDBuilder):
    OUTPUT_FILENAME = "Level-2 DFD - 8.5 survival feature generation.drawio"
    PARENT_LABEL   = "8.5 Survival Feature Generation"
    CONTAINER_X    = 210
    CONTAINER_W    = 990
    CONTAINER_H    = 250
    PAGE_WIDTH     = 1400
    PAGE_HEIGHT    = 900

    def build(self):
        # Row 1: 4 processes with wider boxes and increased spacing
        p1_o, p1_i = self.add_process(
            "8.5.1", "Fit CoxnetSurvivalAnalysis<div>(training data only)</div>",
            20, 40, w=215, h=75,
        )
        p2_o, p2_i = self.add_process(
            "8.5.2", "Compute survival<div>probability S(t)</div><div>at 9 time points</div>",
            260, 40, w=165, h=75,
        )
        p3_o, p3_i = self.add_process(
            "8.5.3", "Compute cumulative<div>hazard H(t)</div><div>at 9 time points</div>",
            450, 40, w=165, h=75,
        )
        p4_o, p4_i = self.add_process(
            "8.5.4", "Compute risk score<div>&amp; expected</div><div>survival time E[T]</div>",
            640, 40, w=180, h=75,
        )

        # Row 2: Concatenate (wider box, centered)
        p5_o, p5_i = self.add_process(
            "8.5.5", "Concatenate with<div>original features</div><div>-&gt; Enhanced dataset</div>",
            260, 155, w=230, h=60,
        )

        # Sequential row-1 edges — labels raised above edge line
        self.add_edge(p1_i, p2_i, "Fitted Cox model", label_y_offset=-12)
        self.add_edge(p2_i, p3_i, "S(t) values",      label_y_offset=-12)
        self.add_edge(p3_i, p4_i, "H(t) values",      label_y_offset=-12)

        # "Scalar features" edge: exit bottom of 8.5.4, enter right of 8.5.5
        # This routes the edge downward first, avoiding the 8.5.3 bounding box
        self.add_edge_styled(
            p4_i, p5_i,
            "Scalar features",
            style_extra="exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=1;entryY=0.5;entryDx=0;entryDy=0;",
            label_y_offset=-12,
        )

        # S(t) and H(t) aggregation edges into 8.5.5 — exit bottom, enter top
        self.add_edge_styled(
            p2_i, p5_i,
            "S(t) at 9 points",
            style_extra="exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.2;entryY=0;entryDx=0;entryDy=0;",
            label_y_offset=-12,
        )
        self.add_edge_styled(
            p3_i, p5_i,
            "H(t) at 9 points",
            style_extra="exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;",
            label_y_offset=-12,
        )

        # Inputs
        self.add_edge(
            self.store_id("D3"), p1_o,
            "df_data_surv<div>(training split only)</div>",
            parent_id="1",
        )
        self.add_edge(
            self.store_id("D4"), p1_o,
            "Best Cox params<div>+ time points</div>",
            parent_id="1",
        )

        # Output to Model Building
        ml_out = self.add_external_entity(
            "8.0 Model Building<div>(Enhanced feature matrix)</div>",
            450, 680, w=200, h=50,
        )
        self.add_edge(
            p5_o, ml_out,
            "Enhanced features<div>(X_surv_train, X_surv_test)</div>",
            parent_id="1",
        )

        print("[8.5 v2] Built Survival Feature Generation sub-DFD: 5 processes (layout fixed)")


if __name__ == "__main__":
    builder = SurvivalFeatureSubDFDv2(DIAGRAMS_DIR)
    builder.build()
    builder.save()
