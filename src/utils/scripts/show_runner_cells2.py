import json

with open("Machine Learning.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
# Show cells 24-33  
for i, cell in enumerate(cells[23:33], start=24):
    src = "".join(cell.get("source", []))
    print(f"\n=== Cell {i+1} ===")
    print(src[:2000])
