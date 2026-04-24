import json

with open("Machine Learning.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Total cells: {len(cells)}")

for i, cell in enumerate(cells):
    src = "".join(cell.get("source", []))
    if "migrate" in src or (src.strip().startswith("from") and src.strip().endswith("import")):
        ctype = cell["cell_type"]
        print(f"\n--- Cell {i+1} (type={ctype}) ---")
        print(repr(src[:300]))
