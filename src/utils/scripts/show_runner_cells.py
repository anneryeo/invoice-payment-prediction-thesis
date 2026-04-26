import json

with open("Machine Learning.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
# Show cells 33-45 
for i, cell in enumerate(cells[32:50], start=33):
    src = "".join(cell.get("source", []))
    print(f"\n=== Cell {i+1} ===")
    print(src[:1200])
    if i+1 >= 45:
        break
