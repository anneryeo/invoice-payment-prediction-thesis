import sqlite3, ast

conn = sqlite3.connect("results/2026_04_18_02/results.db")

# Test two-stage parse
row = conn.execute("SELECT model, parameters FROM experiments WHERE model='two_stage_xgb_ada' LIMIT 1").fetchone()
print("Two-stage model:", row[0])
print("Parameters str:", row[1])

param_str = row[1]
s2_marker = ", stage2="
idx = param_str.index(s2_marker)
stage1 = ast.literal_eval(param_str[len("stage1="):idx])
stage2 = ast.literal_eval(param_str[idx + len(s2_marker):])
print("Parsed stage1:", stage1)
print("Parsed stage2:", stage2)

# Test base model parse
row2 = conn.execute("SELECT model, parameters FROM experiments WHERE model='ada_boost' LIMIT 1").fetchone()
print()
print("Base model:", row2[0])
print("Parameters str:", row2[1])
pairs = ast.literal_eval(row2[1])
print("Parsed dict:", dict(pairs))

# Test ordinal model parse
row3 = conn.execute("SELECT model, parameters FROM experiments WHERE model='ordinal_ada_boost' LIMIT 1").fetchone()
print()
print("Ordinal model:", row3[0])
print("Parameters str:", row3[1])
pairs3 = ast.literal_eval(row3[1])
print("Parsed dict:", dict(pairs3))
