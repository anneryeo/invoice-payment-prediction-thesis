"""
Round 2 layout fix — Level-2 DFD 8.0 Model Building.

Fixes from round1:
- Changed from 3-column horizontal layout to 3-row vertical layout so that
  the three "Fitted …" output edges to D4 exit at distinct y-coordinates,
  eliminating the stacked/overlapping labels on the left side of the diagram.
- Each process now spans the full container width (750 px), giving room for
  all model-name content without overflow.
- Canvas enlarged to 1400×900.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))
from shared.base_sub_dfd_builder import SubDFDBuilder

DIAGRAMS_DIR = Path(__file__).parents[4] / "diagrams"


class ModelBuildingSubDFDv2(SubDFDBuilder):
    OUTPUT_FILENAME = "Level-2 DFD - 8.0 model building.drawio"
    PARENT_LABEL   = "8.0 Model Building"
    CONTAINER_X    = 210
    CONTAINER_W    = 800
    CONTAINER_H    = 390
    PAGE_WIDTH     = 1400
    PAGE_HEIGHT    = 900

    def build(self):
        # Vertical stack: each process spans full width, well-separated y-bands
        p81_o, p81_i = self.add_process(
            "8.1",
            "Base classifier training<div>AdaBoost | Random Forest | XGBoost</div><div>Decision Tree | Gaussian NB | KNN</div>",
            20, 30, w=750, h=90,
        )
        p82_o, p82_i = self.add_process(
            "8.2",
            "Ordinal classifier training<div>Ordinal Ada | Ordinal RF | Ordinal XGB</div><div>(Frank &amp; Hall K-1 binary decomposition)</div>",
            20, 150, w=750, h=90,
        )
        p83_o, p83_i = self.add_process(
            "8.3",
            "Two-stage ensemble training<div>XGB-&gt;XGB | XGB-&gt;RF | RF-&gt;RF</div><div>XGB-&gt;Ada | RF-&gt;Ada | Ada-&gt;XGB</div>",
            20, 270, w=750, h=90,
        )

        # NOTE: 8.2 and 8.3 restricted to tree-based models
        self.add_external_entity(
            "NOTE: 8.2 and 8.3 restricted to tree-based models<div>(XGB, RF, Ada) because SelectFromModel</div>"
            "<div>requires feature_importances_ (MDI).</div><div>KNN and Gaussian NB do not expose this attribute.</div>",
            190, 690, w=440, h=70,
        )

        # Inputs: Baseline training data from D3
        self.add_edge(self.store_id("D3"), p81_o, "Baseline training data", parent_id="1")
        self.add_edge(self.store_id("D3"), p82_o, "Baseline training data", parent_id="1")
        self.add_edge(self.store_id("D3"), p83_o, "Baseline training data", parent_id="1")

        # Outputs to D4 — each process exits at a different y band, so labels are vertically separated
        self.add_edge_styled(
            p81_o, self.store_id("D4"),
            "Fitted base models",
            parent_id="1",
            style_extra="exitX=0;exitY=0.5;exitDx=0;exitDy=0;entryX=1;entryY=0.2;entryDx=0;entryDy=0;",
            label_y_offset=-12,
        )
        self.add_edge_styled(
            p82_o, self.store_id("D4"),
            "Fitted ordinal models",
            parent_id="1",
            style_extra="exitX=0;exitY=0.5;exitDx=0;exitDy=0;entryX=1;entryY=0.5;entryDx=0;entryDy=0;",
            label_y_offset=0,
        )
        self.add_edge_styled(
            p83_o, self.store_id("D4"),
            "Fitted two-stage models",
            parent_id="1",
            style_extra="exitX=0;exitY=0.5;exitDx=0;exitDy=0;entryX=1;entryY=0.8;entryDx=0;entryDy=0;",
            label_y_offset=12,
        )

        # Enhanced features input (from 8.5 Survival Feature Generation)
        enhanced_input = self.add_external_entity(
            "Enhanced feature set<div>(from 8.5 Survival Feature Generation)</div>",
            800, 225, w=200, h=55,
        )
        self.add_edge(enhanced_input, p82_o, "Enhanced features", parent_id="1")

        print("[8.0 v2] Built Model Building sub-DFD: 3 processes, vertical layout (layout fixed)")


if __name__ == "__main__":
    builder = ModelBuildingSubDFDv2(DIAGRAMS_DIR)
    builder.build()
    builder.save()
