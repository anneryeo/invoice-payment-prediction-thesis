"""
Level-1 Modelling DFD revision classes.

D0: Annotation of Model building (8.0) — no visual change, narrative-only marker
D1: Add process 7.5 Cox Survival Analysis Tuning
D2: Add process 8.5 Survival Feature Generation + "Enhanced features" edge to 8.0
D4: Annotate Data preparation (6.0) with balancing strategy + fix "hyperarameters" typo

Key IDs in Level-1 DFD - modelling component.drawio:
  Modelling container:  de6Q2h__l96xfN-PkqQO-16  (x=133, y=390, w=585, h=350)
  6.0 outer:            de6Q2h__l96xfN-PkqQO-17   6.0 inner:  de6Q2h__l96xfN-PkqQO-18
  7.0 outer:            de6Q2h__l96xfN-PkqQO-19   7.0 inner:  de6Q2h__l96xfN-PkqQO-20
  8.0 outer:            de6Q2h__l96xfN-PkqQO-23   8.0 inner:  de6Q2h__l96xfN-PkqQO-24
  9.0 outer:            de6Q2h__l96xfN-PkqQO-31   9.0 inner:  de6Q2h__l96xfN-PkqQO-32
  10.0 outer:           de6Q2h__l96xfN-PkqQO-29   10.0 inner: de6Q2h__l96xfN-PkqQO-30
  11.0 outer:           de6Q2h__l96xfN-PkqQO-64   11.0 inner: de6Q2h__l96xfN-PkqQO-65
  12.0 outer:           de6Q2h__l96xfN-PkqQO-72   12.0 inner: de6Q2h__l96xfN-PkqQO-73
  D2 store container:   de6Q2h__l96xfN-PkqQO-6    (Student information)
  D3 store container:   de6Q2h__l96xfN-PkqQO-8    (Credit sales ledger)
  D4 store container:   de6Q2h__l96xfN-PkqQO-10   (Models)
  GUI:                  de6Q2h__l96xfN-PkqQO-14
  Edge 69 (Model hyperarameters->10.0): de6Q2h__l96xfN-PkqQO-69
  Edge label 69:        b7I2Ov36Tzlnidb0gqtN-1
  Edge 81 (GUI->6.0, unlabeled): de6Q2h__l96xfN-PkqQO-81
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from base_dfd_modifier import DrawioModifier, _new_id, _swimlane_process_xml, _edge_xml
import xml.etree.ElementTree as ET

DRAWIO_PATH = Path(__file__).parents[3] / "diagrams" / "Level-1 DFD - modelling component.drawio"

# Known IDs
MOD_CONTAINER = "de6Q2h__l96xfN-PkqQO-16"
P60_OUTER     = "de6Q2h__l96xfN-PkqQO-17"
P60_INNER     = "de6Q2h__l96xfN-PkqQO-18"
P70_OUTER     = "de6Q2h__l96xfN-PkqQO-19"
P70_INNER     = "de6Q2h__l96xfN-PkqQO-20"
P80_OUTER     = "de6Q2h__l96xfN-PkqQO-23"
P80_INNER     = "de6Q2h__l96xfN-PkqQO-24"
D3_STORE      = "de6Q2h__l96xfN-PkqQO-8"
GUI           = "de6Q2h__l96xfN-PkqQO-14"
EDGE_HYPERPARAM     = "de6Q2h__l96xfN-PkqQO-69"
EDGE_HYPERPARAM_LBL = "b7I2Ov36Tzlnidb0gqtN-1"
EDGE_GUI_TO_P60     = "de6Q2h__l96xfN-PkqQO-81"

# New node IDs (stable, for test assertions)
P75_OUTER = "rev_p75_outer"
P75_INNER = "rev_p75_inner"
P85_OUTER = "rev_p85_outer"
P85_INNER = "rev_p85_inner"


class CoxTuningProcessRevision(DrawioModifier):
    """D1: Add process 7.5 Cox Survival Analysis Tuning between 7.0 and 8.0."""

    def apply(self):
        container = self._id_map.get(MOD_CONTAINER)
        if container is None:
            raise ValueError("Modelling container not found")

        # Expand container to accommodate new process (add 165px width on right)
        geo = container.find("mxGeometry")
        if geo is not None:
            current_w = float(geo.get("width", "585"))
            geo.set("width", str(current_w + 180))

        # Position 7.5 to the right of 8.0 (x=436+100+20=556 within container)
        p75_x, p75_y, p75_w, p75_h = 560, 40, 155, 60
        cells = _swimlane_process_xml(P75_OUTER, "7.5", "Cox Survival<div>Analysis Tuning</div>",
                                      MOD_CONTAINER, p75_x, p75_y, p75_w, p75_h,
                                      fill="#dae8fc", stroke="#6c8ebf")
        # Override inner id so tests can find it
        cells[1].set("id", P75_INNER)
        for c in cells:
            self._xml_root.append(c)
            self._id_map[c.get("id")] = c

        # Edge: 7.0 inner -> 7.5 inner, labeled "Survival analysis data"
        e1 = _edge_xml(_new_id("e_"), P70_INNER, P75_INNER,
                       label="Survival analysis data", parent_id=MOD_CONTAINER)
        # Edge: 7.5 inner -> 8.0 inner, labeled "Tuning parameters"
        e2 = _edge_xml(_new_id("e_"), P75_INNER, P80_INNER,
                       label="Tuning parameters", parent_id=MOD_CONTAINER)
        for cell_list in [e1, e2]:
            for c in cell_list:
                self._xml_root.append(c)
                self._id_map[c.get("id")] = c

        print("  [D1] Added 7.5 Cox Survival Analysis Tuning")


class SurvivalFeatureGenerationRevision(DrawioModifier):
    """D2: Add process 8.5 Survival Feature Generation + Enhanced features edge to 8.0."""

    def apply(self):
        # Position 8.5 below 7.5 in the expanded container
        p85_x, p85_y, p85_w, p85_h = 560, 130, 155, 65
        cells = _swimlane_process_xml(P85_OUTER, "8.5", "Survival Feature<div>Generation</div>",
                                      MOD_CONTAINER, p85_x, p85_y, p85_w, p85_h,
                                      fill="#dae8fc", stroke="#6c8ebf")
        cells[1].set("id", P85_INNER)
        for c in cells:
            self._xml_root.append(c)
            self._id_map[c.get("id")] = c

        # Edge: 7.5 inner -> 8.5 inner, labeled "Tuned Cox model"
        e1 = _edge_xml(_new_id("e_"), P75_INNER, P85_INNER,
                       label="Tuned Cox model", parent_id=MOD_CONTAINER)
        # Edge: 8.5 inner -> 8.0 inner, labeled "Enhanced features"
        e2 = _edge_xml(_new_id("e_"), P85_INNER, P80_INNER,
                       label="Enhanced features", parent_id=MOD_CONTAINER)
        for cell_list in [e1, e2]:
            for c in cell_list:
                self._xml_root.append(c)
                self._id_map[c.get("id")] = c

        print("  [D2] Added 8.5 Survival Feature Generation + Enhanced features edge")


class BalancingStrategyAnnotationRevision(DrawioModifier):
    """D4: Add balancing strategy label to GUI->6.0 edge and fix hyperparameters typo."""

    def apply(self):
        # Fix typo on existing "Model hyperarameters" edge label
        lbl_cell = self._id_map.get(EDGE_HYPERPARAM_LBL)
        if lbl_cell is not None:
            lbl_cell.set("value", "Model hyperparameters")
            print("  [D4] Fixed 'hyperarameters' typo on edge 69 label")

        # Add label to GUI->6.0 edge (currently unlabeled)
        edge_81 = self._id_map.get(EDGE_GUI_TO_P60)
        if edge_81 is not None:
            self.update_or_add_edge_label(EDGE_GUI_TO_P60,
                                          "Model hyperparameters,<div>balancing strategy</div>")
            print("  [D4] Added 'balancing strategy' label to GUI->6.0 edge")
        else:
            print("  [D4] Warning: GUI->6.0 edge not found, skipping label addition")


class ModellingDFDRevisionRunner:
    """Apply all Modelling DFD revisions in sequence to the same file."""

    DRAWIO_PATH = DRAWIO_PATH

    def run(self, output_path: Path | None = None):
        print(f"Loading {self.DRAWIO_PATH}")

        # Apply D1 first (adds 7.5 which D2 depends on)
        d1 = CoxTuningProcessRevision(self.DRAWIO_PATH)
        d1.apply()
        # Save to a temp path then reload for D2 so id_map is fresh
        tmp = self.DRAWIO_PATH.parent / "_tmp_modelling_d1.drawio"
        d1.save(tmp)

        d2 = SurvivalFeatureGenerationRevision(tmp)
        d2.apply()
        tmp2 = self.DRAWIO_PATH.parent / "_tmp_modelling_d2.drawio"
        d2.save(tmp2)

        d4 = BalancingStrategyAnnotationRevision(tmp2)
        d4.apply()

        target = output_path or self.DRAWIO_PATH
        d4.save(target)

        # Clean up temp files
        tmp.unlink(missing_ok=True)
        tmp2.unlink(missing_ok=True)

        print(f"  Saved -> {target}")


if __name__ == "__main__":
    runner = ModellingDFDRevisionRunner()
    runner.run()
