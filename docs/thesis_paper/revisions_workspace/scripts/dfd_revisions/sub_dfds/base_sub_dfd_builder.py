"""
Base class for building new Level-2 DFD drawio files from scratch.

A Level-2 DFD expands a single Level-1 process into its sub-steps,
using the same visual conventions (swimlane containers, data stores,
external entities, and labelled edges).
"""

import uuid
import xml.etree.ElementTree as ET
from pathlib import Path


def _uid(prefix: str = "") -> str:
    return (prefix or "") + uuid.uuid4().hex[:14]


_SWIMLANE_PROCESS = (
    "swimlane;fontStyle=0;childLayout=stackLayout;horizontal=1;startSize=20;"
    "fillColor={fill};horizontalStack=0;resizeParent=1;resizeParentMax=0;"
    "resizeLast=0;collapsible=0;marginBottom=0;swimlaneFillColor=#ffffff;"
    "rounded=1;strokeColor={stroke};"
)

_EDGE_STYLE = (
    "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
    "jettySize=auto;html=1;"
)

_DATA_STORE_OUTER = (
    "shape=table;startSize=30;container=1;collapsible=0;childLayout=tableLayout;"
    "fixedRows=1;rowLines=0;fontStyle=1;align=center;resizeLast=1;"
    "fillColor=#dae8fc;strokeColor=#6c8ebf;"
)

_DATA_STORE_LABEL = (
    "text;html=1;align=left;verticalAlign=middle;resizable=0;"
    "points=[];autosize=1;strokeColor=none;fillColor=none;"
)

_EXTERNAL_ENTITY_STYLE = (
    "rounded=0;whiteSpace=wrap;html=1;"
    "fillColor=#cce5ff;strokeColor=#36393d;"
)

_EDGE_LABEL_STYLE = (
    "edgeLabel;html=1;align=center;verticalAlign=middle;"
    "resizable=0;points=[];"
)

# Shared data store definitions (D1–D5), positioned left of the main swimlane
SHARED_STORES = [
    ("D1", "Raw transaction data",   10, 40),
    ("D2", "Student information",    10, 80),
    ("D3", "Credit sales ledger",    10, 120),
    ("D4", "Models",                 10, 160),
    ("D5", "Influential features",   10, 200),
]


class SubDFDBuilder:
    """Build a new drawio XML file for a Level-2 DFD."""

    # Subclasses must set these
    OUTPUT_FILENAME: str = ""          # filename (no path), e.g. "Level-2 DFD - 5.0 data cleaning.drawio"
    PARENT_LABEL: str = ""             # e.g. "5.0 Data Cleaning"
    CONTAINER_FILL: str = "#cce5ff"
    CONTAINER_X: int = 205
    CONTAINER_Y: int = 260
    CONTAINER_W: int = 620
    CONTAINER_H: int = 320

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._store_ids: dict[str, str] = {}   # "D1" -> outer container id
        self._process_ids: dict[str, str] = {} # "5.1" -> outer id, "5.1_inner" -> inner id

        # Build XML tree skeleton
        self._graph = ET.Element("mxGraphModel", {
            "dx": "1034", "dy": "546", "grid": "1", "gridSize": "10",
            "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
            "fold": "1", "page": "1", "pageScale": "1",
            "pageWidth": "1169", "pageHeight": "827",
            "math": "0", "shadow": "0",
        })
        root = ET.SubElement(self._graph, "root")
        ET.SubElement(root, "mxCell", {"id": "0"})
        ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
        self._root = root

        # Shared layer (data stores, etc.) container
        self._layer_id = "layer_main"
        ET.SubElement(self._root, "mxCell", {
            "id": self._layer_id, "value": "", "style": "swimlane;whiteSpace=wrap;html=1;fillColor=none;strokeColor=none;",
            "parent": "1", "vertex": "1",
        })
        geo = ET.SubElement(self._root[-1], "mxGeometry", {"x": "0", "y": "0", "width": "1169", "height": "827", "as": "geometry"})

        self._add_shared_stores()

    def _add_shared_stores(self):
        for name, label, x, y in SHARED_STORES:
            outer_id = _uid(f"ds_{name}_")
            inner_id = _uid(f"ds_{name}l_")
            outer = ET.Element("mxCell", {
                "id": outer_id, "value": name,
                "style": "shape=table;startSize=20;container=1;collapsible=0;childLayout=tableLayout;"
                         "fixedRows=1;rowLines=0;fontStyle=1;align=center;resizeLast=1;"
                         "fillColor=#dae8fc;strokeColor=#6c8ebf;",
                "parent": "1", "vertex": "1",
            })
            ET.SubElement(outer, "mxGeometry", {"x": str(x), "y": str(y), "width": "170", "height": "30", "as": "geometry"})
            self._root.append(outer)

            inner = ET.Element("mxCell", {
                "id": inner_id, "value": label,
                "style": _DATA_STORE_LABEL,
                "parent": outer_id, "vertex": "1",
            })
            ET.SubElement(inner, "mxGeometry", {"x": "30", "width": "140", "height": "30", "as": "geometry"})
            self._root.append(inner)

            self._store_ids[name] = outer_id
            self._store_ids[f"{name}_inner"] = inner_id

    def _ensure_container(self) -> str:
        """Create the main swimlane container for this sub-DFD if not yet created."""
        if hasattr(self, "_container_id"):
            return self._container_id
        cid = _uid("container_")
        cell = ET.Element("mxCell", {
            "id": cid, "value": self.PARENT_LABEL,
            "style": f"swimlane;whiteSpace=wrap;html=1;fillColor={self.CONTAINER_FILL};strokeColor=#36393d;",
            "parent": "1", "vertex": "1",
        })
        ET.SubElement(cell, "mxGeometry", {
            "x": str(self.CONTAINER_X), "y": str(self.CONTAINER_Y),
            "width": str(self.CONTAINER_W), "height": str(self.CONTAINER_H),
            "as": "geometry",
        })
        self._root.append(cell)
        self._container_id = cid
        return cid

    def add_process(self, number: str, label: str, x: float, y: float,
                    w: float = 120, h: float = 55,
                    fill: str = "#dae8fc", stroke: str = "#6c8ebf") -> tuple[str, str]:
        """Add a process swimlane. Returns (outer_id, inner_id)."""
        container_id = self._ensure_container()
        outer_id = _uid(f"p{number.replace('.','')}_o_")
        inner_id = _uid(f"p{number.replace('.','')}_i_")

        outer = ET.Element("mxCell", {
            "id": outer_id, "value": number,
            "style": _SWIMLANE_PROCESS.format(fill=fill, stroke=stroke),
            "parent": container_id, "vertex": "1",
        })
        ET.SubElement(outer, "mxGeometry", {
            "x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"
        })
        self._root.append(outer)

        inner = ET.Element("mxCell", {
            "id": inner_id, "value": label,
            "style": "text;html=1;align=center;verticalAlign=middle;"
                     "resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;",
            "parent": outer_id, "vertex": "1",
        })
        ET.SubElement(inner, "mxGeometry", {"x": "0", "y": "20", "width": str(w), "height": str(max(h-20, 30)), "as": "geometry"})
        self._root.append(inner)

        self._process_ids[number] = outer_id
        self._process_ids[f"{number}_inner"] = inner_id
        return outer_id, inner_id

    def add_edge(self, source_id: str, target_id: str, label: str = "",
                 parent_id: str | None = None) -> str:
        container_id = self._ensure_container()
        edge_id = _uid("edge_")
        edge = ET.Element("mxCell", {
            "id": edge_id, "value": "",
            "style": _EDGE_STYLE,
            "parent": parent_id or container_id,
            "source": source_id, "target": target_id,
            "edge": "1",
        })
        ET.SubElement(edge, "mxGeometry", {"relative": "1", "as": "geometry"})
        self._root.append(edge)

        if label:
            lbl_id = _uid("elbl_")
            lbl = ET.Element("mxCell", {
                "id": lbl_id, "value": label,
                "style": _EDGE_LABEL_STYLE,
                "parent": edge_id, "vertex": "1", "connectable": "0",
            })
            ET.SubElement(lbl, "mxGeometry", {"x": "0", "y": "0", "relative": "1", "as": "geometry"})
            self._root.append(lbl)

        return edge_id

    def add_external_entity(self, label: str, x: float, y: float,
                             w: float = 150, h: float = 50) -> str:
        eid = _uid("ext_")
        cell = ET.Element("mxCell", {
            "id": eid, "value": label,
            "style": _EXTERNAL_ENTITY_STYLE,
            "parent": "1", "vertex": "1",
        })
        ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
        self._root.append(cell)
        return eid

    def process_id(self, number: str) -> str:
        """Return the outer container id for a process number."""
        return self._process_ids[number]

    def process_inner_id(self, number: str) -> str:
        """Return the inner label cell id for a process number."""
        return self._process_ids[f"{number}_inner"]

    def store_id(self, name: str) -> str:
        return self._store_ids[name]

    def build(self):
        """Override in subclasses to add processes, edges, etc."""
        raise NotImplementedError

    def save(self):
        output_path = self.output_dir / self.OUTPUT_FILENAME
        tree = ET.ElementTree(self._graph)
        ET.indent(tree, space="    ")
        tree.write(str(output_path), encoding="utf-8", xml_declaration=True)
        print(f"  Saved -> {output_path}")
        return output_path
