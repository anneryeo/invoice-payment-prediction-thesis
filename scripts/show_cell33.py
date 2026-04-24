import json

with open("Machine Learning.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
# Show cell 33 full
src = "".join(cells[32].get("source", []))
print("=== Cell 33 FULL ===")
print(src)
