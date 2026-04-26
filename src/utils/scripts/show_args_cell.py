import json

with open("Machine Learning.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
# Show cell 3 full content
src = "".join(cells[2].get("source", []))
print("=== Cell 3 (args) ===")
print(src)
