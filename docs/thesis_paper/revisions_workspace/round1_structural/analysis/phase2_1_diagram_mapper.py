"""
Phase 2.1: Parse the three Level-1 DFD drawio files and produce
diagram_pipeline_map.md — a structured inventory of every DFD element
(processes, data stores, external entities, data flows) cross-referenced
with the corresponding Chapter 3 section.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from html import unescape


DIAGRAMS_DIR = Path(__file__).parents[2] / "diagrams"
CH3_MD = Path(__file__).parents[1] / "outputs" / "chapter3_methodology.md"
OUTPUT = Path(__file__).parents[1] / "outputs" / "diagram_pipeline_map.md"

DFD_FILES = {
    "Processing": DIAGRAMS_DIR / "Level-1 DFD - processing component.drawio",
    "Modelling": DIAGRAMS_DIR / "Level-1 DFD - modelling component.drawio",
    "Analysis": DIAGRAMS_DIR / "Level-1 DFD - analysis component.drawio",
}

# Map process numbers to Chapter 3 sections (derived from manual review)
PROCESS_TO_SECTION = {
    "1.0": "3.2",   # Data importation → Data Acquisition
    "2.0": "3.3",   # Time to payment → Granular Feature Engineering
    "3.0": "3.2",   # Student information → Data Acquisition
    "4.0": "3.3",   # Payment history → Granular Feature Engineering
    "5.0": "3.2",   # Data cleaning → Data Acquisition
    "6.0": "3.6",   # Data preparation → Experimental Design
    "7.0": "3.6",   # Data partitioning → Experimental Design
    "8.0": "3.5",   # Model building → Model Architectures
    "9.0": "3.5",   # Feature selection → Model Architectures
    "10.0": "3.5",  # Model execution → Model Architectures
    "11.0": "3.6",  # Model evaluation → Experimental Design
    "12.0": "3.5",  # Model combination → Model Architectures
    "13.0": "3.5",  # Classify transactions → Model Architectures
    "14.0": "3.6",  # Generate summaries → Experimental Design
    "15.0": "3.6",  # Payment analysis → Experimental Design
    "16.0": "3.6",  # Visualization → Experimental Design
}


def _clean_html(text: str) -> str:
    """Strip HTML tags and unescape entities from drawio cell values."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return " ".join(text.split())


@dataclass
class DFDElement:
    kind: str      # process | data_store | external_entity | data_flow
    node_id: str
    label: str
    number: str = ""
    source_id: str = ""
    target_id: str = ""

    def ch3_section(self) -> str:
        return PROCESS_TO_SECTION.get(self.number, "—")


class DrawioParser:
    """Parse a .drawio XML file and return typed DFD elements."""

    def __init__(self, path: Path):
        tree = ET.parse(str(path))
        self.root = tree.getroot()
        self._cells: dict[str, ET.Element] = {}
        self._elements: list[DFDElement] = []
        self._parse()

    def _parse(self):
        for cell in self.root.iter("mxCell"):
            cell_id = cell.get("id", "")
            self._cells[cell_id] = cell

        for cell in self.root.iter("mxCell"):
            cell_id = cell.get("id", "")
            raw_value = cell.get("value", "")
            if not raw_value or cell_id in ("0", "1"):
                continue
            value = _clean_html(raw_value)
            if not value or len(value) < 2:
                continue

            style = cell.get("style", "")
            source = cell.get("source", "")
            target = cell.get("target", "")

            if source and target:
                self._elements.append(DFDElement("data_flow", cell_id, value,
                                                  source_id=source, target_id=target))
                continue

            if "swimlane" in style and "childLayout=stackLayout" in style:
                number_match = re.match(r"^([\d\.]+)$", value)
                if number_match:
                    num = number_match.group(1)
                    label = self._find_child_label(cell_id)
                    self._elements.append(DFDElement("process", cell_id, label, number=num))
                    continue

            if "shape=mxgraph" in style or ("ellipse" in style and "strokeColor" in style):
                self._elements.append(DFDElement("data_store", cell_id, value))
                continue

            parent = cell.get("parent", "")
            parent_cell = self._cells.get(parent)
            if parent_cell is not None:
                parent_style = parent_cell.get("style", "")
                if "childLayout=stackLayout" in parent_style:
                    continue

            if "rounded=0" in style or "fillColor=#cce5ff" in style:
                if "swimlane" not in style:
                    self._elements.append(DFDElement("external_entity", cell_id, value))
                    continue

            if re.match(r"^D\d", value) or ("text;" in style and "align=left" in style):
                parent = cell.get("parent", "")
                par_cell = self._cells.get(parent)
                if par_cell is not None:
                    child_label = _clean_html(par_cell.get("value", ""))
                    if child_label:
                        self._elements.append(DFDElement("data_store", parent, value))

    def _find_child_label(self, parent_id: str) -> str:
        for cell in self.root.iter("mxCell"):
            if cell.get("parent") == parent_id and cell.get("vertex") == "1":
                val = _clean_html(cell.get("value", ""))
                if val and not re.match(r"^[\d\.]+$", val):
                    return val
        return ""

    def get_processes(self) -> list[DFDElement]:
        return [e for e in self._elements if e.kind == "process" and e.label]

    def get_data_flows(self) -> list[DFDElement]:
        return [e for e in self._elements if e.kind == "data_flow"]

    def get_data_stores(self) -> list[DFDElement]:
        return [e for e in self._elements if e.kind == "data_store"]

    def get_external_entities(self) -> list[DFDElement]:
        return [e for e in self._elements if e.kind == "external_entity"]


class DiagramToMarkdownMapper:
    """Generate a markdown summary of each DFD's elements."""

    def map_all(self, dfd_files: dict[str, Path]) -> str:
        sections = []
        for name, path in dfd_files.items():
            parser = DrawioParser(path)
            sections.append(self._diagram_section(name, parser))
        return "\n\n".join(sections)

    def _diagram_section(self, name: str, parser: DrawioParser) -> str:
        lines = [f"## Level-1 DFD — {name} Component\n"]

        procs = parser.get_processes()
        if procs:
            lines.append("### Processes\n")
            lines.append("| No. | Label | Ch.3 Section |")
            lines.append("|-----|-------|-------------|")
            for p in sorted(procs, key=lambda x: x.number):
                lines.append(f"| {p.number} | {p.label} | {p.ch3_section()} |")
            lines.append("")

        stores = parser.get_data_stores()
        if stores:
            lines.append("### Data Stores\n")
            seen = set()
            for s in stores:
                if s.label not in seen:
                    lines.append(f"- {s.label}")
                    seen.add(s.label)
            lines.append("")

        entities = parser.get_external_entities()
        if entities:
            lines.append("### External Entities\n")
            seen = set()
            for e in entities:
                if e.label not in seen:
                    lines.append(f"- {e.label}")
                    seen.add(e.label)
            lines.append("")

        flows = parser.get_data_flows()
        if flows:
            lines.append("### Data Flows\n")
            seen = set()
            for f in flows:
                if f.label and f.label not in seen:
                    lines.append(f"- {f.label}")
                    seen.add(f.label)
            lines.append("")

        return "\n".join(lines)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    mapper = DiagramToMarkdownMapper()
    content = mapper.map_all(DFD_FILES)
    header = "# Diagram Pipeline Map\n\n"
    header += "_Maps each DFD element (processes, stores, flows) to Chapter 3 sections._\n\n"
    full = header + content
    OUTPUT.write_text(full, encoding="utf-8")
    print(f"Written to {OUTPUT}")

    # Print process inventory per DFD
    for name, path in DFD_FILES.items():
        p = DrawioParser(path)
        procs = p.get_processes()
        print(f"\n{name} DFD processes ({len(procs)}):")
        for proc in sorted(procs, key=lambda x: x.number):
            print(f"  {proc.number}: {proc.label}")


if __name__ == "__main__":
    main()
