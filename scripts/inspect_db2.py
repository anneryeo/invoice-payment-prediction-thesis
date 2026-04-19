import sqlite3
import pandas as pd

DB = "Results/2026_04_18_02/results.db"
conn = sqlite3.connect(DB)

# Distinct models
models = pd.read_sql("SELECT DISTINCT model FROM experiments ORDER BY model", conn)
print("=== MODELS ===")
print(models.to_string(index=False))

# Distinct balance strategies and their counts with threshold
print()
strat = pd.read_sql(
    "SELECT balance_strategy, undersample_threshold, COUNT(*) as cnt FROM experiments GROUP BY balance_strategy, undersample_threshold ORDER BY balance_strategy, undersample_threshold",
    conn
)
print("=== BALANCE STRATEGIES ===")
print(strat.to_string(index=False))

# Distinct phases in metrics
print()
phases = pd.read_sql("SELECT DISTINCT phase FROM metrics", conn)
print("=== METRIC PHASES ===")
print(phases.to_string(index=False))

# Distinct chart types
print()
ctypes = pd.read_sql("SELECT DISTINCT chart_type, phase, COUNT(*) as cnt FROM charts GROUP BY chart_type, phase", conn)
print("=== CHART TYPES ===")
print(ctypes.to_string(index=False))

# Sample metrics row
print()
sample = pd.read_sql("SELECT * FROM metrics LIMIT 3", conn)
print("=== SAMPLE METRICS ===")
print(sample.to_string())

# Sample experiment
print()
exp_sample = pd.read_sql("SELECT id, model, balance_strategy, undersample_threshold, param_hash FROM experiments LIMIT 5", conn)
print("=== SAMPLE EXPERIMENTS ===")
print(exp_sample.to_string())

conn.close()
