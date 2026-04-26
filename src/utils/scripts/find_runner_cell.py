import json

with open("Machine Learning.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
# Show cells 14-35 looking for best_parameters, SurvivalExperimentRunner
for i, cell in enumerate(cells[12:40], start=13):
    src = "".join(cell.get("source", []))
    if "best_param" in src or "SurvivalExperiment" in src or "best_penalty" in src or "tune_cox" in src:
        print(f"\n=== Cell {i+1} ===")
        print(src[:800])
