"""Tests for Level-1 DFD revision scripts (modelling and processing)."""

import sys
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest

DIAGRAMS = Path(__file__).parents[2] / "diagrams"
SCRIPTS  = Path(__file__).parents[1] / "scripts"

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "dfd_revisions"))

MODELLING_DFD  = DIAGRAMS / "Level-1 DFD - modelling component.drawio"
PROCESSING_DFD = DIAGRAMS / "Level-1 DFD - processing component.drawio"


def _parse(path: Path) -> ET.Element:
    return ET.parse(str(path)).getroot()


def _cell_values(root: ET.Element) -> list[str]:
    return [c.get("value", "") for c in root.findall(".//mxCell")]


class TestModellingDFD:
    def test_file_exists(self):
        assert MODELLING_DFD.exists(), f"Missing: {MODELLING_DFD}"

    def test_valid_xml(self):
        _parse(MODELLING_DFD)  # no exception = valid

    def test_process_7_5_present(self):
        root = _parse(MODELLING_DFD)
        values = _cell_values(root)
        assert "7.5" in values, "Process 7.5 not found in modelling DFD"

    def test_process_8_5_present(self):
        root = _parse(MODELLING_DFD)
        values = _cell_values(root)
        assert "8.5" in values, "Process 8.5 not found in modelling DFD"

    def test_cox_label_present(self):
        root = _parse(MODELLING_DFD)
        values = " ".join(_cell_values(root)).lower()
        assert "cox" in values or "survival" in values, (
            "No Cox/Survival label found in modelling DFD after revision"
        )

    def test_balancing_annotation_present(self):
        root = _parse(MODELLING_DFD)
        values = " ".join(_cell_values(root)).lower()
        assert "balancing" in values or "resample" in values, (
            "No balancing strategy annotation found in modelling DFD"
        )

    def test_typo_fix(self):
        root = _parse(MODELLING_DFD)
        all_text = " ".join(_cell_values(root))
        assert "hyperarameters" not in all_text, (
            "Typo 'hyperarameters' still present in modelling DFD"
        )


class TestProcessingDFD:
    def test_file_exists(self):
        assert PROCESSING_DFD.exists(), f"Missing: {PROCESSING_DFD}"

    def test_valid_xml(self):
        _parse(PROCESSING_DFD)

    def test_ml_training_data_edge(self):
        root = _parse(PROCESSING_DFD)
        values = " ".join(_cell_values(root)).lower()
        assert "ml training data" in values, (
            "'ML training data' edge label not found in processing DFD"
        )

    def test_survival_analysis_data_edge(self):
        root = _parse(PROCESSING_DFD)
        values = " ".join(_cell_values(root)).lower()
        assert "survival analysis data" in values, (
            "'Survival analysis data' edge label not found in processing DFD"
        )
