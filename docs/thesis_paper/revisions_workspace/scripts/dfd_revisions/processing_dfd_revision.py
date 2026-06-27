"""
Level-1 Processing DFD revision class.

D3: After Data cleaning (5.0), rename the two outgoing "Cleaned data" edges:
  - xEzRI2CQ2iSpi9w46HNT-74 (5.0 -> GUI) -> "ML training data"
  - xEzRI2CQ2iSpi9w46HNT-78 (5.0 -> D3 Credit sales ledger) -> "Survival analysis data"

Key IDs in Level-1 DFD - processing component.drawio:
  5.0 outer:  xEzRI2CQ2iSpi9w46HNT-65   5.0 inner:  xEzRI2CQ2iSpi9w46HNT-66
  D3 store:   xEzRI2CQ2iSpi9w46HNT-7    (Credit sales ledger)
  GUI:        xEzRI2CQ2iSpi9w46HNT-38
  Edge 74 label (5.0->GUI, "Cleaned data"):          xEzRI2CQ2iSpi9w46HNT-75
  Edge 78 label (5.0->D3, "Cleaned data"):           xEzRI2CQ2iSpi9w46HNT-80
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from base_dfd_modifier import DrawioModifier

DRAWIO_PATH = Path(__file__).parents[3] / "diagrams" / "Level-1 DFD - processing component.drawio"

# IDs of the two edge LABEL cells (children of the respective edge cells)
EDGE_74_LABEL = "xEzRI2CQ2iSpi9w46HNT-75"   # 5.0 -> GUI, label "Cleaned data"
EDGE_78_LABEL = "xEzRI2CQ2iSpi9w46HNT-80"   # 5.0 -> D3, label "Cleaned data"


class DualStreamOutputRevision(DrawioModifier):
    """D3: Rename the two 'Cleaned data' output edges from 5.0 to distinct stream labels."""

    def apply(self):
        # Rename 5.0 -> GUI edge from "Cleaned data" to "ML training data"
        lbl_74 = self._id_map.get(EDGE_74_LABEL)
        if lbl_74 is not None:
            old_val = lbl_74.get("value", "")
            lbl_74.set("value", "ML training data")
            print(f"  [D3] Edge 74 label: {old_val!r} -> 'ML training data'")
        else:
            print("  [D3] Warning: edge 74 label cell not found")

        # Rename 5.0 -> D3 edge from "Cleaned data" to "Survival analysis data"
        lbl_78 = self._id_map.get(EDGE_78_LABEL)
        if lbl_78 is not None:
            old_val = lbl_78.get("value", "")
            lbl_78.set("value", "Survival analysis data")
            print(f"  [D3] Edge 78 label: {old_val!r} -> 'Survival analysis data'")
        else:
            print("  [D3] Warning: edge 78 label cell not found")


class ProcessingDFDRevisionRunner:
    """Apply all Processing DFD revisions."""

    DRAWIO_PATH = DRAWIO_PATH

    def run(self, output_path: Path | None = None):
        print(f"Loading {self.DRAWIO_PATH}")
        rev = DualStreamOutputRevision(self.DRAWIO_PATH)
        rev.apply()
        target = output_path or self.DRAWIO_PATH
        rev.save(target)
        print(f"  Saved -> {target}")


if __name__ == "__main__":
    runner = ProcessingDFDRevisionRunner()
    runner.run()
