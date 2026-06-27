"""
Base class for modifying existing drawio XML files.

Provides methods to add process nodes, edges, and update labels
in an existing Level-1 DFD drawio file.

Drawio XML structure (relevant parts):
  <mxGraphModel>
    <root>
      <mxCell id="0"/>          ← root cell
      <mxCell id="1" parent="0"/> ← default layer
      <mxCell id="..." value="D1" style="shape=..." parent="1" vertex="1">  ← data store container
        <mxCell value="Raw transaction data" style="text;..." parent="..." vertex="1"/>  ← label child
      </mxCell>
      <mxCell id="..." value="Modelling" style="swimlane;..." parent="1" vertex="1">  ← component container
        <mxCell id="..." value="6.0" style="swimlane;...childLayout=stackLayout..." parent="..." vertex="1">  ← process outer
          <mxCell value="Data preparation" style="text;..." parent="..." vertex="1"/>  ← process label
        </mxCell>
        ...
      </mxCell>
      <mxCell style="edgeLabel;..." value="Cleaned data" parent="edge_id"/>  ← edge label
      <mxCell style="..." edge="1" source="id1" target="id2" parent="1"/>    ← edge
    </root>
  </mxGraphModel>
"""

import xml.etree.ElementTree as ET
import uuid
from pathlib import Path
from copy import deepcopy


def _new_id(prefix: str = "") -> str:
    return (prefix or "") + uuid.uuid4().hex[:12]


def _swimlane_process_xml(node_id: str, number: str, label: str, parent_id: str,
                           x: float, y: float, w: float = 120, h: float = 55,
                           fill: str = "#cce5ff", stroke: str = "#36393d") -> list[ET.Element]:
    """Return the two mxCell elements for a stackLayout swimlane process node."""
    outer = ET.Element("mxCell", {
        "id": node_id,
        "value": number,
        "style": (
            f"swimlane;fontStyle=0;childLayout=stackLayout;horizontal=1;startSize=20;"
            f"fillColor={fill};horizontalStack=0;resizeParent=1;resizeParentMax=0;"
            f"resizeLast=0;collapsible=0;marginBottom=0;swimlaneFillColor=#ffffff;"
            f"rounded=1;strokeColor={stroke};"
        ),
        "parent": parent_id,
        "vertex": "1",
    })
    geom = ET.SubElement(outer, "mxGeometry", {"x": str(x), "y": str(y),
                                                "width": str(w), "height": str(h), "as": "geometry"})
    label_id = _new_id("lbl_")
    inner = ET.Element("mxCell", {
        "id": label_id,
        "value": label,
        "style": (
            "text;html=1;align=center;verticalAlign=middle;"
            "resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;"
        ),
        "parent": node_id,
        "vertex": "1",
    })
    inner_geom = ET.SubElement(inner, "mxGeometry", {"x": "0", "y": "20",
                                                       "width": str(w), "height": "30", "as": "geometry"})
    return [outer, inner]


def _edge_xml(edge_id: str, source_id: str, target_id: str,
              label: str = "", parent_id: str = "1") -> list[ET.Element]:
    """Return edge mxCell + optional edge label."""
    edge = ET.Element("mxCell", {
        "id": edge_id,
        "value": "",
        "style": "rounded=0;orthogonalLoop=1;jettySize=auto;exitX=1;exitY=0.5;exitDx=0;"
                 "exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;",
        "parent": parent_id,
        "source": source_id,
        "target": target_id,
        "edge": "1",
    })
    ET.SubElement(edge, "mxGeometry", {"relative": "1", "as": "geometry"})
    elements = [edge]
    if label:
        lbl_id = _new_id("elbl_")
        lbl = ET.Element("mxCell", {
            "id": lbl_id,
            "value": label,
            "style": "edgeLabel;html=1;align=center;verticalAlign=middle;resizable=0;points=[];",
            "parent": edge_id,
            "vertex": "1",
            "connectable": "0",
        })
        ET.SubElement(lbl, "mxGeometry", {"x": "0", "y": "0", "relative": "1", "as": "geometry"})
        elements.append(lbl)
    return elements


class DrawioModifier:
    """Load, modify, and save a drawio XML file."""

    def __init__(self, path: Path):
        self.path = path
        ET.register_namespace("", "")
        self._tree = ET.parse(str(path))
        self._root = self._tree.getroot()
        self._xml_root = self._root.find(".//root")
        if self._xml_root is None:
            raise ValueError(f"No <root> element found in {path}")
        self._id_map: dict[str, ET.Element] = {}
        for cell in self._xml_root.iter("mxCell"):
            cid = cell.get("id", "")
            if cid:
                self._id_map[cid] = cell

    # ── Lookup helpers ──────────────────────────────────────────────────────────

    def find_cell_by_value(self, value_substring: str) -> ET.Element | None:
        for cell in self._xml_root.iter("mxCell"):
            if value_substring.lower() in (cell.get("value") or "").lower():
                return cell
        return None

    def find_cells_by_style(self, style_substring: str) -> list[ET.Element]:
        result = []
        for cell in self._xml_root.iter("mxCell"):
            if style_substring in (cell.get("style") or ""):
                result.append(cell)
        return result

    def find_container_by_value(self, value_substring: str) -> ET.Element | None:
        """Find a swimlane container (parent of all processes in a DFD component)."""
        for cell in self._xml_root.iter("mxCell"):
            style = cell.get("style") or ""
            if "swimlane" in style and "childLayout" not in style:
                val = cell.get("value") or ""
                if value_substring.lower() in val.lower():
                    return cell
        return None

    def find_process_by_number(self, number: str) -> ET.Element | None:
        """Find an outer process swimlane cell by its number value."""
        for cell in self._xml_root.iter("mxCell"):
            style = cell.get("style") or ""
            if "childLayout=stackLayout" in style:
                val = (cell.get("value") or "").strip()
                if val == number:
                    return cell
        return None

    def get_geometry(self, cell: ET.Element) -> dict:
        geo = cell.find("mxGeometry")
        if geo is None:
            return {}
        return {k: float(v) for k, v in geo.attrib.items() if k in ("x", "y", "width", "height")}

    def set_geometry(self, cell: ET.Element, **kwargs):
        geo = cell.find("mxGeometry")
        if geo is None:
            geo = ET.SubElement(cell, "mxGeometry")
        for k, v in kwargs.items():
            geo.set(k, str(v))

    # ── Mutation helpers ────────────────────────────────────────────────────────

    def add_process_node(self, node_id: str, number: str, label: str,
                         parent_id: str, x: float, y: float,
                         w: float = 120, h: float = 55) -> str:
        cells = _swimlane_process_xml(node_id, number, label, parent_id, x, y, w, h)
        for c in cells:
            self._xml_root.append(c)
            self._id_map[c.get("id")] = c
        return node_id

    def add_edge(self, source_id: str, target_id: str,
                 label: str = "", edge_id: str | None = None,
                 parent_id: str = "1") -> str:
        edge_id = edge_id or _new_id("edge_")
        cells = _edge_xml(edge_id, source_id, target_id, label, parent_id)
        for c in cells:
            self._xml_root.append(c)
            self._id_map[c.get("id")] = c
        return edge_id

    def update_cell_value(self, cell: ET.Element, new_value: str):
        cell.set("value", new_value)

    def update_cell_label(self, cell_id: str, new_value: str):
        cell = self._id_map.get(cell_id)
        if cell is None:
            raise KeyError(f"Cell id {cell_id!r} not found")
        cell.set("value", new_value)

    def find_edge_between(self, source_id: str, target_id: str) -> ET.Element | None:
        for cell in self._xml_root.iter("mxCell"):
            if cell.get("source") == source_id and cell.get("target") == target_id:
                return cell
        return None

    def find_outgoing_edges(self, source_id: str) -> list[ET.Element]:
        return [c for c in self._xml_root.iter("mxCell")
                if c.get("source") == source_id]

    def find_incoming_edges(self, target_id: str) -> list[ET.Element]:
        return [c for c in self._xml_root.iter("mxCell")
                if c.get("target") == target_id]

    def find_edge_label(self, edge_id: str) -> ET.Element | None:
        for cell in self._xml_root.iter("mxCell"):
            if cell.get("parent") == edge_id and "edgeLabel" in (cell.get("style") or ""):
                return cell
        return None

    def update_or_add_edge_label(self, edge_id: str, new_label: str):
        lbl = self.find_edge_label(edge_id)
        if lbl is not None:
            lbl.set("value", new_label)
        else:
            lbl_id = _new_id("elbl_")
            lbl = ET.Element("mxCell", {
                "id": lbl_id,
                "value": new_label,
                "style": "edgeLabel;html=1;align=center;verticalAlign=middle;resizable=0;points=[];",
                "parent": edge_id,
                "vertex": "1",
                "connectable": "0",
            })
            ET.SubElement(lbl, "mxGeometry", {"x": "0", "y": "0", "relative": "1", "as": "geometry"})
            self._xml_root.append(lbl)
            self._id_map[lbl_id] = lbl

    # ── Persistence ─────────────────────────────────────────────────────────────

    def save(self, output_path: Path | None = None):
        target = output_path or self.path
        ET.indent(self._tree, space="    ")
        self._tree.write(str(target), encoding="utf-8", xml_declaration=True)

    def save_inplace(self):
        self.save(self.path)
