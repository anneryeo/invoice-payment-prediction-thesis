# # Chart Loading Diagnostic
# Tests `ResultsRepository` end-to-end: load → hydrate → plot.
# Run each cell in order. The goal is to confirm whether chart data
# is actually present in the database and deserializes correctly.

import sys
import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Point this at your project root so local imports resolve ─────────────────
PROJECT_ROOT = r"."
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DB_PATH = r"Results\2026_03_19_01\results.db"

print("Imports OK")

# ## 1. Raw DB inspection — what's actually stored?

con = sqlite3.connect(DB_PATH)

# Which tables exist?
tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables:", tables)

# How many rows in each key table?
for t in ["experiments", "metrics", "charts", "features"]:
    if t in tables:
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {n} rows")

con.close()

con = sqlite3.connect(DB_PATH)

# Inspect one charts row — is data NULL or populated?
rows = con.execute(
    "SELECT experiment_id, phase, chart_type, "
    "CASE WHEN data IS NULL THEN 'NULL' "
    "     WHEN data = '' THEN 'EMPTY STRING' "
    "     ELSE 'OK (' || LENGTH(data) || ' chars)' END AS status "
    "FROM charts LIMIT 12"
).fetchall()

print(f"{'exp_id':>8}  {'phase':10}  {'chart_type':20}  status")
print("-" * 60)
for r in rows:
    print(f"{r[0]:>8}  {r[1]:10}  {r[2]:20}  {r[3]}")

con.close()

con = sqlite3.connect(DB_PATH)

# Peek at the raw JSON for experiment_id=1, baseline roc_curve
row = con.execute(
    "SELECT data FROM charts WHERE experiment_id=1 AND phase='baseline' AND chart_type='roc_curve'"
).fetchone()

if row is None:
    print("ROW NOT FOUND — chart was never written to the DB")
elif row[0] is None:
    print("ROW EXISTS but data is NULL")
else:
    print(f"Data length: {len(row[0])} chars")
    print(f"First 300 chars: {row[0][:300]}")

con.close()

# ## 2. ResultsRepository load — does hydration work?

from src.modules.machine_learning.utils.io.results_repository import ResultsRepository

repo   = ResultsRepository(DB_PATH)
models = repo.load_models_dict()

print(f"Loaded {len(models)} model entries")

# Pick the first key for all subsequent cells
sample_key = next(iter(models))
print(f"Sample key: {sample_key}")

entry = models[sample_key]
print("\nBefore hydration:")
print("  baseline roc_curve:", entry['baseline']['evaluation']['charts']['roc_curve'])
print("  enhanced roc_curve:", entry['enhanced']['evaluation']['charts']['roc_curve'])

repo.hydrate_model_charts(entry)

print("After hydration:")
for phase in ("baseline", "enhanced"):
    charts = entry[phase]['evaluation']['charts']
    for name, val in charts.items():
        if val is None:
            status = "None  ← PROBLEM"
        elif isinstance(val, dict):
            status = f"dict with {len(val)} keys: {list(val.keys())[:5]}"
        elif isinstance(val, list):
            status = f"list of {len(val)} items"
        else:
            status = f"{type(val).__name__}: {str(val)[:60]}"
        print(f"  {phase:10}  {name:20}  {status}")

print("\nFeatures after hydration:")
for phase in ("baseline", "enhanced"):
    print(f"  {phase}: {entry[phase]['features'][:5]} ... ({len(entry[phase]['features'])} total)")
    print(f"  {phase} weights: {str(entry[phase].get('weights', 'MISSING'))[:80]}")

# ## 3. Plot all four charts for the sample model

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

def plot_roc(ax, roc: dict, title="ROC Curve"):
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", linewidth=1)
    if not roc:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="red")
        return
    for i, (cls, entry) in enumerate(sorted(roc.items(), key=lambda x: x[0])):
        fpr = entry.get("fpr", []) if isinstance(entry, dict) else []
        tpr = entry.get("tpr", []) if isinstance(entry, dict) else []
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], linewidth=2, label=f"Class {cls}")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)


def plot_pr(ax, pr: dict, title="Precision-Recall"):
    ax.set_title(title, fontsize=10, fontweight="bold")
    if not pr:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="red")
        return
    for i, (cls, entry) in enumerate(sorted(pr.items(), key=lambda x: x[0])):
        recall    = entry.get("recall",    []) if isinstance(entry, dict) else []
        precision = entry.get("precision", []) if isinstance(entry, dict) else []
        ax.plot(recall, precision, color=COLORS[i % len(COLORS)], linewidth=2, label=f"Class {cls}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)


def plot_cm(ax, cm_raw, title="Confusion Matrix"):
    ax.set_title(title, fontsize=10, fontweight="bold")
    if cm_raw is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="red")
        return
    cm = np.array(cm_raw)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    n = cm.shape[0]
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            total = cm[i].sum()
            pct   = cm[i, j] / total * 100 if total > 0 else 0
            ax.text(j, i, f"{cm[i,j]}\n{pct:.1f}%",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))


def plot_features(ax, features: list, weights, title="Feature Importance"):
    ax.set_title(title, fontsize=10, fontweight="bold")
    if not features:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="red")
        return
    if weights and isinstance(weights, dict):
        importance = [float(weights.get(f, 0)) for f in features]
    elif weights and isinstance(weights, list):
        importance = [float(w) for w in weights]
    else:
        importance = list(range(len(features), 0, -1))  # rank-based fallback

    paired = sorted(zip(importance, features), key=lambda p: abs(p[0]), reverse=True)[:15]
    imp_vals = [p[0] for p in paired]
    feat_names = [p[1] for p in paired]

    colors = [COLORS[0] if i == 0 else "#93c5fd" for i in range(len(imp_vals))]
    ax.barh(feat_names[::-1], [abs(v) for v in imp_vals[::-1]], color=colors[::-1])
    ax.set_xlabel("Importance (absolute)")
    ax.tick_params(axis="y", labelsize=7)


print("Plot helpers defined")

for phase in ("baseline", "enhanced"):
    charts   = entry[phase]["evaluation"]["charts"]
    features = entry[phase]["features"]
    weights  = entry[phase].get("weights")

    roc = charts.get("roc_curve")      or {}
    pr  = charts.get("pr_curve")       or {}
    cm  = charts.get("confusion_matrix")

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(
        f"{entry['model']}  ·  {entry['balance_strategy']}  ·  {phase.upper()}",
        fontsize=12, fontweight="bold"
    )
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.4)

    plot_roc(fig.add_subplot(gs[0]), roc,      f"ROC  ({phase})")
    plot_pr( fig.add_subplot(gs[1]), pr,       f"PR   ({phase})")
    plot_cm( fig.add_subplot(gs[2]), cm,       f"CM   ({phase})")
    plot_features(fig.add_subplot(gs[3]), features, weights, f"Features ({phase})")

    plt.show()

# ## 4. If charts show 'No data' above — inspect the raw JSON structure

# Run this if roc / pr / cm above showed 'No data' or looked wrong.
# Prints the exact structure coming out of the DB so we can see what keys exist.

con = sqlite3.connect(DB_PATH)
sample_exp_id = entry["experiment_id"]

for phase in ("baseline", "enhanced"):
    for chart_type in ("roc_curve", "pr_curve", "confusion_matrix"):
        row = con.execute(
            "SELECT data FROM charts WHERE experiment_id=? AND phase=? AND chart_type=?",
            (sample_exp_id, phase, chart_type)
        ).fetchone()

        print(f"\n{'='*60}")
        print(f"{phase} / {chart_type}  (experiment_id={sample_exp_id})")
        if row is None:
            print("  → ROW MISSING from charts table")
        elif row[0] is None:
            print("  → data column is NULL")
        else:
            try:
                parsed = json.loads(row[0])
                if isinstance(parsed, dict):
                    print(f"  → dict, top-level keys: {list(parsed.keys())}")
                    first_key = next(iter(parsed))
                    print(f"  → first entry ({first_key!r}): {str(parsed[first_key])[:200]}")
                elif isinstance(parsed, list):
                    print(f"  → list of {len(parsed)} items, first: {str(parsed[0])[:200]}")
                else:
                    print(f"  → {type(parsed).__name__}: {str(parsed)[:200]}")
            except json.JSONDecodeError as e:
                print(f"  → JSON PARSE ERROR: {e}")
                print(f"  → Raw (first 200 chars): {row[0][:200]}")

con.close()

