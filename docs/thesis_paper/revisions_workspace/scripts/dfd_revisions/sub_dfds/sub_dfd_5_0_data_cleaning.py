"""Level-2 DFD — 5.0 Data Cleaning sub-processes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from dfd_revisions.sub_dfds.base_sub_dfd_builder import SubDFDBuilder

DIAGRAMS_DIR = Path(__file__).parents[4] / "diagrams"


class DataCleaningSubDFD(SubDFDBuilder):
    OUTPUT_FILENAME = "Level-2 DFD - 5.0 data cleaning.drawio"
    PARENT_LABEL   = "5.0 Data Cleaning"
    CONTAINER_W    = 680
    CONTAINER_H    = 220

    def build(self):
        # 5.1: Invoice Building (InvoiceBuilder) — left
        p51_o, p51_i = self.add_process(
            "5.1",
            "Invoice building<div>(InvoiceBuilder)</div><div>discount, payment allocation, ThreadPool</div>",
            20, 40, w=180, h=80,
        )
        # 5.2: Feature Engineering — center
        p52_o, p52_i = self.add_process(
            "5.2",
            "Feature engineering<div>(FeatureEngineer)</div><div>DTP lags, payment trends, plan encoding</div>",
            230, 40, w=180, h=80,
        )
        # 5.3: Post-processing — right
        p53_o, p53_i = self.add_process(
            "5.3",
            "Post-processing<div>(InvoicePostProcessor)</div><div>year filter, winsorization, column drop</div>",
            440, 40, w=190, h=80,
        )
        # 5.4: Data stream split — bottom center
        p54_o, p54_i = self.add_process(
            "5.4",
            "Data stream split",
            270, 155, w=150, h=45,
        )

        self.add_edge(p51_i, p52_i, "Allocated invoices")
        self.add_edge(p52_i, p53_i, "Engineered features")
        self.add_edge(p53_i, p54_i, "Filtered dataset")

        # 5.4 outputs two streams
        gui = self.add_external_entity("ML Pipeline<div>(Modelling DFD)</div>", 130, 660, w=160, h=50)
        cox = self.add_external_entity("Cox Tuning<div>(Process 7.5)</div>", 580, 660, w=160, h=50)
        self.add_edge(p54_i, gui, "ML training data<div>(df_data, censor==1)</div>", parent_id="1")
        self.add_edge(p54_i, cox, "Survival analysis data<div>(df_data_surv, full)</div>", parent_id="1")

        # Input from D1
        self.add_edge(self.store_id("D1"), p51_o, "Raw transactions", parent_id="1")
        self.add_edge(self.store_id("D2"), p52_o, "Student demographics", parent_id="1")

        print("[5.0] Built Data Cleaning sub-DFD: 4 processes + 2 output streams")


if __name__ == "__main__":
    builder = DataCleaningSubDFD(DIAGRAMS_DIR)
    builder.build()
    builder.save()
