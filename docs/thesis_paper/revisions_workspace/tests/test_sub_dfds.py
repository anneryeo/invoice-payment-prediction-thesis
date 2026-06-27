"""Tests for generated Level-2 sub-DFD drawio files."""

import xml.etree.ElementTree as ET
from pathlib import Path
import pytest

DIAGRAMS = Path(__file__).parents[2] / "diagrams"


def _parse(path: Path) -> ET.Element:
    return ET.parse(str(path)).getroot()


def _swimlane_values(root: ET.Element) -> list[str]:
    """Return cell values for process swimlane containers."""
    return [
        c.get("value", "")
        for c in root.findall(".//mxCell")
        if "childLayout=stackLayout" in (c.get("style") or "")
    ]


def _edge_count(root: ET.Element) -> int:
    return sum(1 for c in root.findall(".//mxCell") if c.get("edge") == "1")


def _all_values(root: ET.Element) -> str:
    return " ".join(c.get("value", "") for c in root.findall(".//mxCell")).lower()


class TestDataImportationDFD:
    PATH = DIAGRAMS / "Level-2 DFD - 1.0 data importation.drawio"

    def test_file_exists(self):
        assert self.PATH.exists(), f"Missing: {self.PATH}"

    def test_valid_xml(self):
        _parse(self.PATH)

    def test_four_processes(self):
        root = _parse(self.PATH)
        procs = _swimlane_values(root)
        assert "1.1" in procs and "1.2" in procs and "1.3" in procs and "1.4" in procs

    def test_has_edges(self):
        root = _parse(self.PATH)
        assert _edge_count(root) >= 3

    def test_contains_excel_reference(self):
        root = _parse(self.PATH)
        assert "excel" in _all_values(root) or "xlsx" in _all_values(root)


class TestDataCleaningDFD:
    PATH = DIAGRAMS / "Level-2 DFD - 5.0 data cleaning.drawio"

    def test_file_exists(self):
        assert self.PATH.exists()

    def test_valid_xml(self):
        _parse(self.PATH)

    def test_four_processes(self):
        root = _parse(self.PATH)
        procs = _swimlane_values(root)
        assert all(p in procs for p in ["5.1", "5.2", "5.3", "5.4"])

    def test_dual_stream_labels(self):
        root = _parse(self.PATH)
        vals = _all_values(root)
        assert "ml training data" in vals or "df_data" in vals
        assert "survival" in vals


class TestCoxTuningDFD:
    PATH = DIAGRAMS / "Level-2 DFD - 7.5 cox survival analysis tuning.drawio"

    def test_file_exists(self):
        assert self.PATH.exists()

    def test_valid_xml(self):
        _parse(self.PATH)

    def test_six_processes(self):
        root = _parse(self.PATH)
        procs = _swimlane_values(root)
        expected = ["7.5.1", "7.5.2", "7.5.3", "7.5.4", "7.5.5", "7.5.6"]
        assert all(p in procs for p in expected), f"Missing processes. Found: {procs}"

    def test_c_index_reference(self):
        root = _parse(self.PATH)
        assert "c-index" in _all_values(root) or "concordance" in _all_values(root)

    def test_time_points_reference(self):
        root = _parse(self.PATH)
        assert "time point" in _all_values(root)

    def test_has_edges(self):
        root = _parse(self.PATH)
        assert _edge_count(root) >= 5


class TestModelBuildingDFD:
    PATH = DIAGRAMS / "Level-2 DFD - 8.0 model building.drawio"

    def test_file_exists(self):
        assert self.PATH.exists()

    def test_valid_xml(self):
        _parse(self.PATH)

    def test_three_processes(self):
        root = _parse(self.PATH)
        procs = _swimlane_values(root)
        assert all(p in procs for p in ["8.1", "8.2", "8.3"])

    def test_mentions_ordinal(self):
        root = _parse(self.PATH)
        assert "ordinal" in _all_values(root)

    def test_mentions_two_stage(self):
        root = _parse(self.PATH)
        assert "two-stage" in _all_values(root) or "two_stage" in _all_values(root)


class TestSurvivalFeatureDFD:
    PATH = DIAGRAMS / "Level-2 DFD - 8.5 survival feature generation.drawio"

    def test_file_exists(self):
        assert self.PATH.exists()

    def test_valid_xml(self):
        _parse(self.PATH)

    def test_five_processes(self):
        root = _parse(self.PATH)
        procs = _swimlane_values(root)
        expected = ["8.5.1", "8.5.2", "8.5.3", "8.5.4", "8.5.5"]
        assert all(p in procs for p in expected), f"Missing processes. Found: {procs}"

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
        assert _edge_count(root) >= 5
