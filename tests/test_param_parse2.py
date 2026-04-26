import sqlite3, ast

conn = sqlite3.connect("results/2026_04_18_02/results.db")

rows = conn.execute("SELECT model, parameters FROM experiments WHERE model='two_stage_xgb_ada' LIMIT 1").fetchall()
s = rows[0][1]
print("type:", type(s))
print("repr:", repr(s[:120]))
print("first char:", repr(s[0]))
print("last char:", repr(s[-1]))

# Try stripping outer quotes
s2 = s.strip('"').strip("'")
print("\nAfter strip:", repr(s2[:80]))

# Now test parsing
s2_marker = ", stage2="
idx = s2.index(s2_marker)
stage1 = ast.literal_eval(s2[len("stage1="):idx])
stage2 = ast.literal_eval(s2[idx + len(s2_marker):])
print("Parsed stage1:", stage1)
print("Parsed stage2:", stage2)
