"""Tests for Phase 2: diagram mapper and pipeline tracer outputs."""

import pytest
from pathlib import Path

OUTPUTS = Path(__file__).parents[1] / "outputs"
DIAG_MD = OUTPUTS / "diagram_pipeline_map.md"
PIPE_MD = OUTPUTS / "code_pipeline_trace.md"


class TestDiagramMap:
    def test_file_exists(self):
        assert DIAG_MD.exists(), f"Missing: {DIAG_MD}"

    def test_contains_processing_processes(self):
        content = DIAG_MD.read_text(encoding="utf-8")
        for proc in ["1.0", "2.0", "3.0", "4.0", "5.0"]:
            assert proc in content, f"Processing process {proc} not found"

    def test_contains_modelling_processes(self):
        content = DIAG_MD.read_text(encoding="utf-8")
        for proc in ["6.0", "7.0", "8.0", "9.0", "10.0", "11.0", "12.0"]:
            assert proc in content, f"Modelling process {proc} not found"

    def test_contains_analysis_processes(self):
        content = DIAG_MD.read_text(encoding="utf-8")
        for proc in ["13.0", "14.0", "15.0", "16.0"]:
            assert proc in content, f"Analysis process {proc} not found"


class TestPipelineTrace:
    def test_file_exists(self):
        assert PIPE_MD.exists(), f"Missing: {PIPE_MD}"

    def test_contains_key_steps(self):
        content = PIPE_MD.read_text(encoding="utf-8").lower()
        for term in ["clean_datasets", "cox", "datapreparer", "survival"]:
            assert term in content, f"Expected term '{term}' not in pipeline trace"

    def test_contains_new_steps_flagged(self):
        content = PIPE_MD.read_text(encoding="utf-8")
        assert "NEW" in content or "missing" in content.lower(), (
            "Expected NEW/missing flag for steps 7.5 and 8.5"
        )

    def test_has_at_least_10_steps(self):
        content = PIPE_MD.read_text(encoding="utf-8")
        # Each step starts with "## step" (two hashes, lowercase)
        step_count = content.lower().count("## step")
        assert step_count >= 10, f"Expected >= 10 steps, got {step_count}"
