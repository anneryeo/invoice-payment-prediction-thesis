import sqlite3
import pandas as pd
import json

DB = "Results/2026_04_18_02/results.db"
conn = sqlite3.connect(DB)

# Check features table structure
feat = pd.read_sql("SELECT * FROM features LIMIT 2", conn)
print("=== FEATURES SAMPLE ===")
for _, row in feat.iterrows():
    print(f"  exp_id={row['experiment_id']} phase={row['phase']} method={row['feature_method']}")
    print(f"  feature_parameters: {row['feature_parameters'][:100] if row['feature_parameters'] else 'None'}")
    fj = json.loads(row['features_json']) if row['features_json'] else []
    wj = json.loads(row['weights_json']) if row['weights_json'] else {}
    print(f"  features_json (count={len(fj)}): {fj[:5]}")
    print(f"  weights_json (count={len(wj)}): {list(wj.items())[:5]}")
    print()

# Chart data sample
print("=== CHART DATA SAMPLE (roc_curve) ===")
chart = pd.read_sql("SELECT * FROM charts WHERE chart_type='roc_curve' LIMIT 1", conn)
for _, row in chart.iterrows():
    data = json.loads(row['data'])
    print(f"  exp_id={row['experiment_id']} phase={row['phase']} chart_type={row['chart_type']}")
    print(f"  data keys: {list(data.keys())[:10]}")
    if 'fpr' in data:
        print(f"  fpr type: {type(data['fpr'])}, sample: {str(data['fpr'])[:80]}")
    if 'tpr' in data:
        print(f"  tpr type: {type(data['tpr'])}, sample: {str(data['tpr'])[:80]}")
    if 'auc' in data:
        print(f"  auc: {data['auc']}")

print()
print("=== CHART DATA SAMPLE (confusion_matrix) ===")
chart = pd.read_sql("SELECT * FROM charts WHERE chart_type='confusion_matrix' LIMIT 1", conn)
for _, row in chart.iterrows():
    data = json.loads(row['data'])
    print(f"  keys: {list(data.keys())}")
    if 'matrix' in data:
        print(f"  matrix: {data['matrix']}")
    if 'labels' in data:
        print(f"  labels: {data['labels']}")

print()
print("=== METADATA ===")
meta = pd.read_sql("SELECT * FROM metadata", conn)
for _, row in meta.iterrows():
    d = json.loads(row['data'])
    print(json.dumps(d, indent=2)[:800])

conn.close()
