"""Level-2 DFD — 8.0 Model Building sub-processes.

Three sub-processes:
  8.1 Base classifier training (6 models)
  8.2 Ordinal classifier training (3 models, Frank & Hall K-1 decomposition)
  8.3 Two-stage ensemble training (6 combinations)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from dfd_revisions.sub_dfds.base_sub_dfd_builder import SubDFDBuilder

DIAGRAMS_DIR = Path(__file__).parents[4] / "diagrams"


class ModelBuildingSubDFD(SubDFDBuilder):
    OUTPUT_FILENAME = "Level-2 DFD - 8.0 model building.drawio"
    PARENT_LABEL   = "8.0 Model Building"
    CONTAINER_W    = 780
    CONTAINER_H    = 280

    def build(self):
        # 8.1: Base classifiers (left column)
        p81_o, p81_i = self.add_process(
            "8.1",
            "Base classifier training<div>AdaBoost | Random Forest | XGBoost</div><div>Decision Tree | Gaussian NB | KNN</div>",
            20, 40, w=230, h=80,
        )
        # 8.2: Ordinal classifiers (center column)
        p82_o, p82_i = self.add_process(
            "8.2",
            "Ordinal classifier training<div>Ordinal Ada | Ordinal RF | Ordinal XGB</div><div>(Frank &amp; Hall K-1 binary decomposition)</div>",
            280, 40, w=220, h=80,
        )
        # 8.3: Two-stage ensembles (right column)
        p83_o, p83_i = self.add_process(
            "8.3",
            "Two-stage ensemble training<div>XGB-&gt;XGB | XGB-&gt;RF | RF-&gt;RF</div><div>XGB-&gt;Ada | RF-&gt;Ada | Ada-&gt;XGB</div>",
            530, 40, w=225, h=80,
        )

        # Note on tree-based restriction
        note_id = self.add_external_entity(
            "NOTE: 8.2 and 8.3 restricted to tree-based models<div>(XGB, RF, Ada) because SelectFromModel</div><div>requires feature_importances_ (MDI).</div><div>KNN and Gaussian NB do not expose this attribute.</div>",
            190, 630, w=440, h=70,
        )

        # All three receive Training data (baseline) + Enhanced features
        self.add_edge(self.store_id("D3"), p81_o, "Baseline training data", parent_id="1")
        self.add_edge(self.store_id("D3"), p82_o, "Baseline training data", parent_id="1")
        self.add_edge(self.store_id("D3"), p83_o, "Baseline training data", parent_id="1")

        # Output: trained model weights to D4
        self.add_edge(p81_o, self.store_id("D4"), "Fitted base models", parent_id="1")
        self.add_edge(p82_o, self.store_id("D4"), "Fitted ordinal models", parent_id="1")
        self.add_edge(p83_o, self.store_id("D4"), "Fitted two-stage models", parent_id="1")

        # Note on enhanced features path
        enhanced_input = self.add_external_entity(
            "Enhanced feature set<div>(from 8.5 Survival Feature Generation)</div>",
            650, 230, w=200, h=55,
        )
        self.add_edge(enhanced_input, p81_o, "Enhanced features", parent_id="1")

        print("[8.0] Built Model Building sub-DFD: 3 processes (6+3+6 model variants)")


if __name__ == "__main__":
    builder = ModelBuildingSubDFD(DIAGRAMS_DIR)
    builder.build()
    builder.save()
