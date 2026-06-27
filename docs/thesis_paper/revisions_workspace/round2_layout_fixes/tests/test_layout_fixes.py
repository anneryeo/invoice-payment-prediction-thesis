"""Tests for Round 2 layout-fixed Level-2 sub-DFD drawio files."""

import xml.etree.ElementTree as ET
from pathlib import Path
import pytest

DIAGRAMS = Path(__file__).parents[3] / "diagrams"


def _parse(path: Path) -> ET.Element:
    return ET.parse(str(path)).getroot()


def _swimlane_values(root: ET.Element) -> list[str]:
    return [
        c.get("value", "")
        for c in root.findall(".//mxCell")
        if "childLayout=stackLayout" in (c.get("style") or "")
    ]


def _all_values(root: ET.Element) -> str:
    return " ".join(c.get("value", "") for c in root.findall(".//mxCell")).lower()


def _edge_count(root: ET.Element) -> int:
    return sum(1 for c in root.findall(".//mxCell") if c.get("edge") == "1")


def _page_dimensions(root: ET.Element) -> tuple[int, int]:
    graph = root if root.tag == "mxGraphModel" else root.find(".//mxGraphModel")
    if graph is None:
        return 0, 0
    return int(graph.get("pageWidth", 0)), int(graph.get("pageHeight", 0))


def _get_process_geometry(root: ET.Element, number: str) -> dict:
    """Return geometry dict for a process swimlane by its number value."""
    for cell in root.findall(".//mxCell"):
        style = cell.get("style") or ""
        if "childLayout=stackLayout" in style and cell.get("value") == number:
            geo = cell.find("mxGeometry")
            if geo is not None:
                return {k: float(v) for k, v in geo.attrib.items() if k in ("x", "y", "width", "height")}
    return {}


def _has_label_y_offset(root: ET.Element) -> bool:
    """Return True if any edge label has a non-zero y offset in its geometry."""
    for cell in root.findall(".//mxCell"):
        if "edgeLabel" in (cell.get("style") or ""):
            geo = cell.find("mxGeometry")
            if geo is not None and float(geo.get("y", 0)) != 0:
                return True
    return False


# ─── 7.5 Cox Tuning ──────────────────────────────────────────────────────────

class TestCoxTuningLayoutFix:
    PATH = DIAGRAMS / "Level-2 DFD - 7.5 cox survival analysis tuning.drawio"

    def test_file_exists(self):
        assert self.PATH.exists()

    def test_valid_xml(self):
        _parse(self.PATH)

    def test_canvas_enlarged(self):
        root = _parse(self.PATH)
        w, h = _page_dimensions(root)
        assert w >= 1400 and h >= 900, f"Expected 1400×900 canvas, got {w}×{h}"

    def test_six_processes(self):
        root = _parse(self.PATH)
        procs = _swimlane_values(root)
        assert all(p in procs for p in ["7.5.1", "7.5.2", "7.5.3", "7.5.4", "7.5.5", "7.5.6"])

    def test_process_753_wide_enough(self):
        root = _parse(self.PATH)
        geo = _get_process_geometry(root, "7.5.3")
        assert geo.get("width", 0) >= 190, \
            f"7.5.3 width {geo.get('width')} too narrow for 'CoxnetSurvivalAnalysis'"

    def test_process_754_wide_enough(self):
        root = _parse(self.PATH)
        geo = _get_process_geometry(root, "7.5.4")
        assert geo.get("width", 0) >= 170, \
            f"7.5.4 width {geo.get('width')} too narrow for 'Harrell concordance'"

    def test_edge_labels_have_y_offset(self):
        root = _parse(self.PATH)
        assert _has_label_y_offset(root), "No edge label y-offsets found; labels may overlap edges"

    def test_has_edges(self):
        root = _parse(self.PATH)
        assert _edge_count(root) >= 7

    def test_c_index_reference(self):
        root = _parse(self.PATH)
        assert "c-index" in _all_values(root) or "concordance" in _all_values(root)


# ─── 8.0 Model Building ──────────────────────────────────────────────────────

class TestModelBuildingLayoutFix:
    PATH = DIAGRAMS / "Level-2 DFD - 8.0 model building.drawio"

    def test_file_exists(self):
        assert self.PATH.exists()

    def test_valid_xml(self):
        _parse(self.PATH)

    def test_canvas_enlarged(self):
        root = _parse(self.PATH)
        w, h = _page_dimensions(root)
        assert w >= 1400 and h >= 900

    def test_three_processes(self):
        root = _parse(self.PATH)
        procs = _swimlane_values(root)
        assert all(p in procs for p in ["8.1", "8.2", "8.3"])

    def test_processes_vertically_stacked(self):
        root = _parse(self.PATH)
        g81 = _get_process_geometry(root, "8.1")
        g82 = _get_process_geometry(root, "8.2")
        g83 = _get_process_geometry(root, "8.3")
        # In vertical layout, y-values must be strictly increasing with at least 80 px gap
        assert g82.get("y", 0) > g81.get("y", 0) + 80, "8.1 and 8.2 not vertically separated"
        assert g83.get("y", 0) > g82.get("y", 0) + 80, "8.2 and 8.3 not vertically separated"

    def test_output_edges_use_staggered_exit(self):
        root = _parse(self.PATH)
        has_stagger = any(
            "exitY=0.2" in (c.get("style") or "") or
            "exitY=0.5" in (c.get("style") or "") or
            "exitY=0.8" in (c.get("style") or "")
            for c in root.findall(".//mxCell")
            if c.get("edge") == "1"
        )
        assert has_stagger, "No staggered exit styles found on output edges"

    def test_mentions_ordinal(self):
        root = _parse(self.PATH)
        assert "ordinal" in _all_values(root)

    def test_mentions_two_stage(self):
        root = _parse(self.PATH)
        assert "two-stage" in _all_values(root) or "two_stage" in _all_values(root)


# ─── 8.5 Survival Feature Generation ─────────────────────────────────────────

class TestSurvivalFeatureLayoutFix:
    PATH = DIAGRAMS / "Level-2 DFD - 8.5 survival feature generation.drawio"

    def test_file_exists(self):
        assert self.PATH.exists()

    def test_valid_xml(self):
        _parse(self.PATH)

    def test_canvas_enlarged(self):
        root = _parse(self.PATH)
        w, h = _page_dimensions(root)
        assert w >= 1400 and h >= 900

    def test_five_processes(self):
        root = _parse(self.PATH)
        procs = _swimlane_values(root)
        assert all(p in procs for p in ["8.5.1", "8.5.2", "8.5.3", "8.5.4", "8.5.5"])

    def test_process_851_wide_enough(self):
        root = _parse(self.PATH)
        geo = _get_process_geometry(root, "8.5.1")
        assert geo.get("width", 0) >= 200, \
            f"8.5.1 width {geo.get('width')} too narrow for 'CoxnetSurvivalAnalysis'"

    def test_scalar_features_edge_has_exit_bottom(self):
        root = _parse(self.PATH)
        has_bottom_exit = any(
            "exitY=1" in (c.get("style") or "")
            for c in root.findall(".//mxCell")
            if c.get("edge") == "1"
        )
        assert has_bottom_exit, "No bottom-exit edges found; Scalar features edge may still cross boxes"

    def test_edge_labels_have_y_offset(self):
        root = _parse(self.PATH)
        assert _has_label_y_offset(root), "No edge label y-offsets found"

    def test_st_and_ht_labels(self):
        root = _parse(self.PATH)
        vals = _all_values(root)
        assert "s(t)" in vals or "survival probability" in vals
        assert "h(t)" in vals or "cumulative hazard" in vals

    def test_enhanced_output(self):
        root = _parse(self.PATH)
        assert "enhanced" in _all_values(root)

    def test_has_edges(self):
        root = _parse(self.PATH)
        assert _edge_count(root) >= 7
