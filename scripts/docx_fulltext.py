"""Dump full text of every block in the thesis docx with indices."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

DOC = r"D:\Developer\Projects\THESIS-Utilizing-ML-to-Solve-the-IPPP\Beley-Reyes_Thesis2-ACM.docx"

def iter_block_items(parent):
    for child in parent.element.body.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)

doc = Document(DOC)
for i, block in enumerate(iter_block_items(doc)):
    if isinstance(block, Table):
        print(f"=== [{i}] TABLE ===")
        for r in block.rows:
            print("  ROW: " + " | ".join(c.text.strip() for c in r.cells))
    else:
        text = block.text
        if not text.strip():
            continue
        style = block.style.name if block.style else "?"
        print(f"=== [{i}] ({style}) ===")
        print(text)
