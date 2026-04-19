import sqlite3
import pandas as pd

DB = "results/2026_04_18_02/results.db"
conn = sqlite3.connect(DB)

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("=== TABLES ===")
for (t,) in tables:
    print(f"  {t}")

print()
for (t,) in tables:
    cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
    count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"--- {t} ({count} rows) ---")
    for col in cols:
        print(f"  {col[1]:40s} {col[2]}")
    print()

conn.close()
