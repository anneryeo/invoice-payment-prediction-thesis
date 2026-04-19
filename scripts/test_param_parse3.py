import sqlite3, ast

conn = sqlite3.connect("Results/2026_04_18_02/results.db")

# Test base model
rows = conn.execute("SELECT model, parameters FROM experiments WHERE model='ada_boost' LIMIT 1").fetchall()
s = rows[0][1]
print("Base model repr:", repr(s[:80]))

# Test ordinal model
rows3 = conn.execute("SELECT model, parameters FROM experiments WHERE model='ordinal_ada_boost' LIMIT 1").fetchall()
s3 = rows3[0][1]
print("Ordinal repr:", repr(s3[:80]))
