"""Dump the structural outline of the thesis docx: headings, captions, tables, figures."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

DOC = r"D:\Developer\Projects\THESIS-Utilizing-ML-to-Solve-the-IPPP\Beley-Reyes_Thesis2-ACM.docx"

def iter_block_items(parent):
    from docx.oxml.ns import qn
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)

def has_image(par):
    return bool(par._p.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip'))

doc = Document(DOC)
for i, block in enumerate(iter_block_items(doc)):
    if isinstance(block, Table):
        rows = len(block.rows)
        cols = len(block.columns)
        first = " | ".join(c.text.strip()[:30] for c in block.rows[0].cells) if rows else ""
        print(f"[{i}] TABLE {rows}x{cols} :: {first[:120]}")
    else:
        style = block.style.name if block.style else "?"
        text = block.text.strip()
        img = " <IMG>" if has_image(block) else ""
        if not text and not img:
            continue
        flag = ""
        if style.lower().startswith("heading") or "title" in style.lower():
            flag = "## "
        print(f"[{i}] ({style}){img} {flag}{text[:160]}")
