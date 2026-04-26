import json

with open("Machine Learning.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Total cells: {len(cells)}")

# Look for args/settings cells
for i, cell in enumerate(cells[:20]):  # First 20 cells
    src = "".join(cell.get("source", []))
    if "args" in src or "target_feature" in src or "parameters_dir" in src or "SimpleNamespace" in src:
        print(f"\n--- Cell {i+1} ---")
        print(src[:600])
