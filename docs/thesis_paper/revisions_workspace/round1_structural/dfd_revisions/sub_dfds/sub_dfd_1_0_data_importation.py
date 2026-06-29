"""Level-2 DFD — 1.0 Data Importation sub-processes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))
from shared.base_sub_dfd_builder import SubDFDBuilder

DIAGRAMS_DIR = Path(__file__).parents[4] / "diagrams"


class DataImportationSubDFD(SubDFDBuilder):
    OUTPUT_FILENAME = "Level-2 DFD - 1.0 data importation.drawio"
    PARENT_LABEL   = "1.0 Data Importation"
    CONTAINER_W    = 640
    CONTAINER_H    = 180

    def build(self):
        # Row: 1.1 -> 1.2 -> 1.3 -> 1.4
        p11_o, p11_i = self.add_process("1.1", "Load Excel files<div>(async, parallel)</div>", 20, 40, w=140, h=65)
        p12_o, p12_i = self.add_process("1.2", "Data type<div>conversion</div>", 190, 40, w=120, h=65)
        p13_o, p13_i = self.add_process("1.3", "Datetime<div>parsing</div>", 340, 40, w=120, h=65)
        p14_o, p14_i = self.add_process("1.4", "Lookup-based<div>due date updates</div>", 490, 40, w=130, h=65)

        self.add_edge(p11_i, p12_i, "Raw DataFrames")
        self.add_edge(p12_i, p13_i, "Typed columns")
        self.add_edge(p13_i, p14_i, "Parsed dates")

        # Inputs from D1 store
        self.add_edge(self.store_id("D1"), p11_o, "Revenue Ledger (.xlsx)", parent_id="1")
        self.add_edge(self.store_id("D2"), p11_o, "Enrollee Information (.xlsx)", parent_id="1")

        print(f"[1.0] Built Data Importation sub-DFD: 4 processes")


if __name__ == "__main__":
    builder = DataImportationSubDFD(DIAGRAMS_DIR)
    builder.build()
    builder.save()
