"""Tests for Phase 1: Chapter 3 docx-to-markdown extraction."""

import sys
from pathlib import Path
import pytest

OUTPUTS = Path(__file__).parents[1] / "outputs"
CH3_MD  = OUTPUTS / "chapter3_methodology.md"


def test_file_exists():
    assert CH3_MD.exists(), f"Missing: {CH3_MD}"


def test_file_non_empty():
    content = CH3_MD.read_text(encoding="utf-8")
    assert len(content.strip()) > 100


def test_contains_chapter3_heading():
    content = CH3_MD.read_text(encoding="utf-8").lower()
    assert "chapter 3" in content or "methodology" in content


def test_contains_dfd_reference():
    content = CH3_MD.read_text(encoding="utf-8").lower()
    assert "data flow" in content or "dfd" in content or "level-1" in content


def test_contains_section_headings():
    content = CH3_MD.read_text(encoding="utf-8")
    headings = [ln for ln in content.splitlines() if ln.startswith("#")]
    assert len(headings) >= 3, f"Expected at least 3 headings, got {len(headings)}"
