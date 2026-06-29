"""Level-2 DFD — 8.5 Survival Feature Generation sub-processes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))
from shared.base_sub_dfd_builder import SubDFDBuilder

DIAGRAMS_DIR = Path(__file__).parents[4] / "diagrams"


class SurvivalFeatureSubDFD(SubDFDBuilder):
    OUTPUT_FILENAME = "Level-2 DFD - 8.5 survival feature generation.drawio"
    PARENT_LABEL   = "8.5 Survival Feature Generation"
    CONTAINER_W    = 800
    CONTAINER_H    = 210

    def build(self):
        # Row 1: Fit -> Compute S(t) -> Compute H(t)
        p1_o, p1_i = self.add_process(
            "8.5.1", "Fit CoxnetSurvivalAnalysis<div>(training data only)</div>",
            20, 40, w=170, h=65,
        )
        p2_o, p2_i = self.add_process(
            "8.5.2", "Compute survival<div>probability S(t)</div><div>at 9 time points</div>",
            220, 40, w=145, h=65,
        )
        p3_o, p3_i = self.add_process(
            "8.5.3", "Compute cumulative<div>hazard H(t)</div><div>at 9 time points</div>",
            395, 40, w=145, h=65,
        )
        p4_o, p4_i = self.add_process(
            "8.5.4", "Compute risk score<div>&amp; expected</div><div>survival time E[T]</div>",
            570, 40, w=145, h=65,
        )
        # Row 2: Concatenate
        p5_o, p5_i = self.add_process(
            "8.5.5", "Concatenate with<div>original features</div><div>-&gt; Enhanced dataset</div>",
            300, 140, w=195, h=55,
        )

        self.add_edge(p1_i, p2_i, "Fitted Cox model")
        self.add_edge(p2_i, p3_i, "S(t) values")
        self.add_edge(p3_i, p4_i, "H(t) values")
        self.add_edge(p4_i, p5_i, "Scalar features")
        self.add_edge(p2_i, p5_i, "S(t) at 9 points")
        self.add_edge(p3_i, p5_i, "H(t) at 9 points")

        # Inputs
        self.add_edge(self.store_id("D3"), p1_o, "df_data_surv<div>(training split only)</div>", parent_id="1")
        self.add_edge(self.store_id("D4"), p1_o, "Best Cox params<div>+ time points</div>", parent_id="1")

        # Output to ML pipeline
        ml_out = self.add_external_entity(
            "8.0 Model Building<div>(Enhanced feature matrix)</div>",
            450, 640, w=200, h=50,
        )
        self.add_edge(p5_o, ml_out, "Enhanced features<div>(X_surv_train, X_surv_test)</div>", parent_id="1")

        print("[8.5] Built Survival Feature Generation sub-DFD: 5 processes")


if __name__ == "__main__":
    builder = SurvivalFeatureSubDFD(DIAGRAMS_DIR)
    builder.build()
    builder.save()
