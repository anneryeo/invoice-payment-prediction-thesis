import sqlite3
import pandas as pd
import json

DB = "results/2026_04_18_02/results.db"
conn = sqlite3.connect(DB)

# ROC curve structure
print("=== CHART DATA SAMPLE (roc_curve) ===")
chart = pd.read_sql("SELECT * FROM charts WHERE chart_type='roc_curve' LIMIT 1", conn)
for _, row in chart.iterrows():
    data = json.loads(row['data'])
    print(f"  exp_id={row['experiment_id']} phase={row['phase']}")
    print(f"  type={type(data)}, keys={list(data.keys()) if isinstance(data, dict) else 'list'}")
    if isinstance(data, dict):
        for k, v in list(data.items())[:3]:
            print(f"    [{k}]: {str(v)[:120]}")

print()
print("=== CHART DATA SAMPLE (confusion_matrix) ===")
chart = pd.read_sql("SELECT * FROM charts WHERE chart_type='confusion_matrix' LIMIT 1", conn)
for _, row in chart.iterrows():
    data = json.loads(row['data'])
    print(f"  exp_id={row['experiment_id']} phase={row['phase']}")
    print(f"  type={type(data)}")
    print(f"  sample: {str(data)[:400]}")

print()
print("=== CHART DATA SAMPLE (pr_curve) ===")
chart = pd.read_sql("SELECT * FROM charts WHERE chart_type='pr_curve' LIMIT 1", conn)
for _, row in chart.iterrows():
    data = json.loads(row['data'])
    print(f"  exp_id={row['experiment_id']} phase={row['phase']}")
    print(f"  type={type(data)}")
    if isinstance(data, dict):
        for k, v in list(data.items())[:3]:
            print(f"    [{k}]: {str(v)[:120]}")

print()
print("=== METADATA ===")
meta = pd.read_sql("SELECT * FROM metadata", conn)
for _, row in meta.iterrows():
    d = json.loads(row['data'])
    print(json.dumps(d, indent=2)[:1000])

print()
print("=== CLASS MAPPINGS ===")
cm = pd.read_sql("SELECT * FROM class_mappings", conn)
for _, row in cm.iterrows():
    d = json.loads(row['data'])
    print(json.dumps(d, indent=2)[:500])

conn.close()
