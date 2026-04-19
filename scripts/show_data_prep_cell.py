import json

with open("Machine Learning.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
# Find cell with 'CreditSalesProcessor' and 'best_parameters'
for i, cell in enumerate(cells):
    src = "".join(cell.get("source", []))
    if "best_parameters" in src or "best_penalty" in src or "CreditSalesProcessor" in src:
        print(f"\n=== Cell {i+1} (id={cell.get('id','')}) ===")
        print(src[:3000])
