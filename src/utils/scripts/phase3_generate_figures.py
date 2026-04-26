"""
Quick validation: run analysis cells 1-10 (no McNemar's re-run) to verify
figures are generated correctly before running the full notebook.
"""
import sqlite3
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH    = "results/2026_04_18_02/results.db"
OUTPUT_DIR = Path("docs/202616APRIL-RESULTSGRAPHS")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 120, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})

BASE_MODELS    = ["ada_boost", "decision_tree", "gaussian_naive_bayes", "knn", "random_forest", "xgboost"]
ORDINAL_MODELS = ["ordinal_ada_boost", "ordinal_random_forest", "ordinal_xgboost"]
TWO_STAGE_MODELS = ["two_stage_ada_xgb", "two_stage_rf_ada", "two_stage_rf_rf",
                    "two_stage_xgb_ada", "two_stage_xgb_rf", "two_stage_xgb_xgb"]
ALL_MODELS = BASE_MODELS + ORDINAL_MODELS + TWO_STAGE_MODELS

MODEL_DISPLAY = {
    "ada_boost": "AdaBoost", "decision_tree": "Decision Tree",
    "gaussian_naive_bayes": "Gaussian NB", "knn": "KNN",
    "random_forest": "Random Forest", "xgboost": "XGBoost",
    "ordinal_ada_boost": "Ordinal AdaBoost", "ordinal_random_forest": "Ordinal RF",
    "ordinal_xgboost": "Ordinal XGBoost", "two_stage_ada_xgb": "TS Ada->XGB",
    "two_stage_rf_ada": "TS RF->Ada", "two_stage_rf_rf": "TS RF->RF",
    "two_stage_xgb_ada": "TS XGB->Ada", "two_stage_xgb_rf": "TS XGB->RF",
    "two_stage_xgb_xgb": "TS XGB->XGB",
}
FAMILY_MAP = {}
FAMILY_MAP.update({m: "Base"      for m in BASE_MODELS})
FAMILY_MAP.update({m: "Ordinal"   for m in ORDINAL_MODELS})
FAMILY_MAP.update({m: "Two-Stage" for m in TWO_STAGE_MODELS})

ORDINAL_BASE_MAP = {
    "ordinal_ada_boost": "ada_boost", "ordinal_random_forest": "random_forest",
    "ordinal_xgboost": "xgboost",
}
TWO_STAGE_BASE_MAP = {
    "two_stage_ada_xgb": "xgboost", "two_stage_rf_ada": "ada_boost",
    "two_stage_rf_rf": "random_forest", "two_stage_xgb_ada": "ada_boost",
    "two_stage_xgb_rf": "random_forest", "two_stage_xgb_xgb": "xgboost",
}
STRATEGY_ORDER  = ["none", "smote", "borderline_smote", "smote_tomek",
                   "hybrid@0.5", "hybrid@0.7", "hybrid@0.9"]
STRATEGY_LABELS = {"none": "None", "smote": "SMOTE",
                   "borderline_smote": "Borderline SMOTE",
                   "smote_tomek": "SMOTE+Tomek",
                   "hybrid@0.5": "Hybrid @0.5",
                   "hybrid@0.7": "Hybrid @0.7",
                   "hybrid@0.9": "Hybrid @0.9"}
CLASS_NAMES     = ["On-Time (0)", "30-Day (1)", "60-Day (2)", "90-Day (3)"]
FAMILY_PALETTE  = {"Base": "#4878CF", "Ordinal": "#6ACC65", "Two-Stage": "#D65F5F"}
STRATEGY_PALETTE = dict(zip(STRATEGY_ORDER, sns.color_palette("tab10", 7)))
SURVIVAL_FEATURES = {
    "survival_prob", "hazard", "expected_survival", "partial_hazard", "log_partial_hazard",
    "survival_at_30", "survival_at_60", "survival_at_90",
}

print("[S1] Loading database...")
conn = sqlite3.connect(DB_PATH)
exp_df  = pd.read_sql("SELECT * FROM experiments", conn)
met_df  = pd.read_sql("SELECT * FROM metrics",     conn)
feat_df = pd.read_sql("SELECT * FROM features",    conn)
class_mappings = json.loads(pd.read_sql("SELECT data FROM class_mappings",   conn).iloc[0, 0])
survival_data  = json.loads(pd.read_sql("SELECT data FROM survival_results", conn).iloc[0, 0])
metadata       = json.loads(pd.read_sql("SELECT data FROM metadata",         conn).iloc[0, 0])

met_wide = met_df.pivot_table(
    index="experiment_id", columns="phase",
    values=["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro"],
    aggfunc="first",
)
met_wide.columns = [f"{phase}_{metric}" for metric, phase in met_wide.columns]
met_wide = met_wide.reset_index()

df = exp_df.merge(met_wide, left_on="id", right_on="experiment_id", how="left")
df["strategy_label"] = df.apply(
    lambda r: f"hybrid@{r['undersample_threshold']:.1f}"
              if r["balance_strategy"] == "hybrid" and pd.notna(r["undersample_threshold"])
              else r["balance_strategy"],
    axis=1,
)
df["model_display"] = df["model"].map(MODEL_DISPLAY)
df["family"]        = df["model"].map(FAMILY_MAP)
df["f1_lift"]       = df["enhanced_f1_macro"] - df["baseline_f1_macro"]
print(f"  Loaded {len(df)} experiments")

# ── S2: Quality checks ────────────────────────────────────────────────────────
print("[S2] Quality check...")
for col in ["enhanced_f1_macro", "enhanced_roc_auc_macro"]:
    n = df[col].isna().sum()
    print(f"  {col}: {n} NULLs {'OK' if n == 0 else 'ISSUE'}")
hybrid_nulls = df[df["balance_strategy"] == "hybrid"]["undersample_threshold"].isna().sum()
print(f"  Hybrid threshold NULLs: {hybrid_nulls} {'OK' if hybrid_nulls == 0 else 'ISSUE'}")
strat_counts = df.groupby("strategy_label").size()
for s in STRATEGY_ORDER:
    cnt = strat_counts.get(s, 0)
    print(f"  {s:<22}: {cnt:>4} {'OK' if cnt == 156 else 'ISSUE'}")

# ── S3: Rankings ──────────────────────────────────────────────────────────────
print("[S3] Model rankings...")
rank_f1 = df.sort_values("enhanced_f1_macro", ascending=False).head(5)
for i, (_, r) in enumerate(rank_f1.iterrows(), 1):
    print(f"  #{i}: {r['model_display']:<25} {r['strategy_label']:<18} F1={r['enhanced_f1_macro']:.4f} AUC={r['enhanced_roc_auc_macro']:.4f}")

rho, pval = stats.spearmanr(df["enhanced_f1_macro"].dropna(), df["enhanced_roc_auc_macro"].dropna())
print(f"  Spearman r(F1,AUC)={rho:.4f} p={pval:.2e}")

# ── S4: Best per model ────────────────────────────────────────────────────────
print("[S4] Best per model type...")
best_per_model = df.loc[df.groupby("model")["enhanced_f1_macro"].idxmax()]
for _, r in best_per_model.sort_values("enhanced_f1_macro", ascending=False).iterrows():
    print(f"  {r['model_display']:<25} {r['strategy_label']:<18} F1={r['enhanced_f1_macro']:.4f}")

# ── S5: Ordinal/TS lift + figure ─────────────────────────────────────────────
print("[S5] Computing ordinal/TS lift...")
def _best(model_name, metric="enhanced_f1_macro"):
    sub = df[df["model"] == model_name]
    return sub[metric].max() if not sub.empty else np.nan

cmp_rows = []
for ord_m, base_m in ORDINAL_BASE_MAP.items():
    for met in ("enhanced_f1_macro", "enhanced_roc_auc_macro"):
        cmp_rows.append({"variant": MODEL_DISPLAY[ord_m], "base": MODEL_DISPLAY[base_m],
                         "type": "Ordinal", "metric": met.replace("enhanced_","").replace("_macro",""),
                         "delta": _best(ord_m, met) - _best(base_m, met)})
for ts_m, base_m in TWO_STAGE_BASE_MAP.items():
    for met in ("enhanced_f1_macro", "enhanced_roc_auc_macro"):
        cmp_rows.append({"variant": MODEL_DISPLAY[ts_m], "base": MODEL_DISPLAY[base_m],
                         "type": "Two-Stage", "metric": met.replace("enhanced_","").replace("_macro",""),
                         "delta": _best(ts_m, met) - _best(base_m, met)})
cmp_df = pd.DataFrame(cmp_rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, met_short, ylabel in zip(axes, ["f1","roc_auc"], ["Δ F1 Macro","Δ AUC Macro"]):
    sub = cmp_df[cmp_df["metric"] == met_short].sort_values("delta")
    colors = [FAMILY_PALETTE["Ordinal"] if t == "Ordinal" else FAMILY_PALETTE["Two-Stage"] for t in sub["type"]]
    bars = ax.barh(sub["variant"], sub["delta"], color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(ylabel); ax.set_title(f"Lift over base ({ylabel})")
    for bar, val in zip(bars, sub["delta"]):
        x = bar.get_width()
        ax.text(x + 0.001*(1 if x>=0 else -1), bar.get_y()+bar.get_height()/2,
                f"{val:+.3f}", va="center", ha="left" if x>=0 else "right", fontsize=7)
patches = [mpatches.Patch(color=FAMILY_PALETTE["Ordinal"], label="Ordinal"),
           mpatches.Patch(color=FAMILY_PALETTE["Two-Stage"], label="Two-Stage")]
axes[0].legend(handles=patches, loc="lower right")
plt.suptitle("Ordinal & Two-Stage Variants vs Base Classifiers", fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_ordinal_twostage_lift.png")
plt.close()
print("  Saved fig_ordinal_twostage_lift.png")

# ── S6: F1/AUC bar charts ─────────────────────────────────────────────────────
print("[S6] Generating F1/AUC bar charts...")
agg = df.groupby(["model", "strategy_label"])[
    ["enhanced_f1_macro","enhanced_roc_auc_macro","baseline_f1_macro","baseline_roc_auc_macro"]
].max().reset_index()

def _bar_chart(metric_col, ylabel, title_suffix, fname):
    n_m  = len(ALL_MODELS)
    n_s  = len(STRATEGY_ORDER)
    x    = np.arange(n_m)
    w    = 0.10
    offs = np.linspace(-(n_s-1)/2*w, (n_s-1)/2*w, n_s)
    fig, ax = plt.subplots(figsize=(16, 5))
    for i, strat in enumerate(STRATEGY_ORDER):
        vals = [agg[(agg["model"]==m)&(agg["strategy_label"]==strat)][metric_col].values[0]
                if len(agg[(agg["model"]==m)&(agg["strategy_label"]==strat)]) else np.nan
                for m in ALL_MODELS]
        ax.bar(x+offs[i], vals, w, label=STRATEGY_LABELS[strat],
               color=STRATEGY_PALETTE[strat], edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m,m) for m in ALL_MODELS], rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel(ylabel); ax.set_title(f"{ylabel} by Model & Balance Strategy ({title_suffix})")
    ax.legend(ncol=4, fontsize=8, loc="upper right"); ax.set_ylim(0, 1.05)
    for xp in [len(BASE_MODELS)-0.5, len(BASE_MODELS)+len(ORDINAL_MODELS)-0.5]:
        ax.axvline(xp, color="gray", linewidth=0.8, linestyle=":")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")

_bar_chart("enhanced_f1_macro",      "F1 Macro",  "enhanced features", "fig_f1_by_strategy.png")
_bar_chart("enhanced_roc_auc_macro", "AUC Macro", "enhanced features", "fig_auc_by_strategy.png")
_bar_chart("baseline_f1_macro",      "F1 Macro",  "baseline features", "fig_f1_baseline_by_strategy.png")

# ── S7: Heatmaps ─────────────────────────────────────────────────────────────
print("[S7] Generating heatmaps...")
def _heatmap(metric_col, title, fname):
    pivot = (agg.pivot_table(index="strategy_label", columns="model",
                              values=metric_col, aggfunc="max")
               .reindex(index=STRATEGY_ORDER, columns=ALL_MODELS)
               .rename(columns=MODEL_DISPLAY).rename(index=STRATEGY_LABELS))
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.4, linecolor="white",
                vmin=pivot.values[~np.isnan(pivot.values)].min()*0.97,
                annot_kws={"size": 7.5})
    ax.set_title(title, fontweight="bold"); ax.set_xlabel(""); ax.set_ylabel("Balance Strategy")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8.5)
    ax.axvline(len(BASE_MODELS), color="black", linewidth=1.5)
    ax.axvline(len(BASE_MODELS)+len(ORDINAL_MODELS), color="black", linewidth=1.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")

_heatmap("enhanced_f1_macro",      "F1 Macro (enhanced) — Model × Balance Strategy",  "fig_heatmap_f1_enhanced.png")
_heatmap("baseline_f1_macro",      "F1 Macro (baseline) — Model × Balance Strategy",  "fig_heatmap_f1_baseline.png")
_heatmap("enhanced_roc_auc_macro", "AUC Macro (enhanced) — Model × Balance Strategy", "fig_heatmap_auc_enhanced.png")

# ── S8: Confusion matrices for top-3 ─────────────────────────────────────────
print("[S8] Plotting confusion matrices...")
top3_rows = df.sort_values("enhanced_f1_macro", ascending=False).head(3).reset_index(drop=True)
top3_ids  = top3_rows["id"].tolist()
ph = ",".join(["?"]*len(top3_ids))
cm_data = pd.read_sql(
    f"SELECT experiment_id, data FROM charts WHERE chart_type='confusion_matrix' AND phase='enhanced' AND experiment_id IN ({ph})",
    conn, params=top3_ids,
)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ai, (_, row_info) in enumerate(top3_rows.iterrows()):
    ax = axes[ai]
    cm_row = cm_data[cm_data["experiment_id"] == row_info["id"]]
    if cm_row.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); continue
    cm_raw = np.array(json.loads(cm_row.iloc[0]["data"]), dtype=float)
    cm_norm = cm_raw / cm_raw.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=7.5)
    ax.set_yticklabels(CLASS_NAMES, fontsize=7.5)
    ax.set_xlabel("Predicted", fontsize=8.5); ax.set_ylabel("Actual", fontsize=8.5)
    ax.set_title(f"#{ai+1} {row_info['model_display']}\n{row_info['strategy_label']} F1={row_info['enhanced_f1_macro']:.3f}",
                 fontsize=9, fontweight="bold")
    for i in range(4):
        for j in range(4):
            pct = cm_norm[i, j]; count = int(cm_raw[i, j])
            ax.text(j, i, f"{pct:.2f}\n({count})", ha="center", va="center",
                    color="white" if pct > 0.55 else "black", fontsize=7)
plt.suptitle("Confusion Matrices — Top-3 Models (enhanced features)", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_confusion_matrices_top3.png")
plt.close()
print("  Saved fig_confusion_matrices_top3.png")

# ── S9: ROC curves ────────────────────────────────────────────────────────────
print("[S9] Plotting ROC curves...")
top5_auc = df.sort_values("enhanced_roc_auc_macro", ascending=False).head(5).reset_index(drop=True)
top5_ids = top5_auc["id"].tolist()
ph = ",".join(["?"]*len(top5_ids))
roc_data = pd.read_sql(
    f"SELECT experiment_id, data FROM charts WHERE chart_type='roc_curve' AND phase='enhanced' AND experiment_id IN ({ph})",
    conn, params=top5_ids,
)
FPR_GRID    = np.linspace(0, 1, 500)
LINE_STYLES = ["-","--","-.",":", (0,(3,1,1,1))]
LINE_COLORS = sns.color_palette("tab10", 5)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax in axes:
    ax.plot([0,1],[0,1],"k--",linewidth=0.8,label="Random (AUC=0.50)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")

for idx, (_, rrow) in enumerate(top5_auc.iterrows()):
    roc_row = roc_data[roc_data["experiment_id"] == rrow["id"]]
    if roc_row.empty: continue
    roc_d = json.loads(roc_row.iloc[0]["data"])
    tpr_interps = [np.interp(FPR_GRID, np.array(roc_d[str(k)]["fpr"]), np.array(roc_d[str(k)]["tpr"]))
                   for k in range(4) if str(k) in roc_d]
    macro_tpr = np.mean(tpr_interps, axis=0)
    label = f"{rrow['model_display']} / {rrow['strategy_label']} (AUC={rrow['enhanced_roc_auc_macro']:.4f})"
    axes[0].plot(FPR_GRID, macro_tpr, linestyle=LINE_STYLES[idx], color=LINE_COLORS[idx],
                 linewidth=1.8, label=label)
axes[0].legend(fontsize=7, loc="lower right")
axes[0].set_title("Macro-Avg OvR ROC (Top-5 models)")

best_id  = top5_auc.loc[0, "id"]
roc_row  = roc_data[roc_data["experiment_id"] == best_id]
if not roc_row.empty:
    roc_d = json.loads(roc_row.iloc[0]["data"])
    cls_colors = sns.color_palette("Set1", 4)
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        k = str(cls_idx)
        if k not in roc_d: continue
        fpr_a = np.array(roc_d[k]["fpr"]); tpr_a = np.array(roc_d[k]["tpr"])
        cls_auc = roc_d[k].get("auc", float(np.trapezoid(tpr_a, fpr_a)))
        axes[1].plot(fpr_a, tpr_a, color=cls_colors[cls_idx], linewidth=1.8,
                     label=f"{cls_name} (AUC={cls_auc:.3f})")
axes[1].legend(fontsize=8, loc="lower right")
axes[1].set_title(f"Per-Class ROC — {top5_auc.loc[0,'model_display']}")

plt.suptitle("ROC Curves (enhanced features)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_roc_curves.png")
plt.close()
print("  Saved fig_roc_curves.png")

# ── S10: Feature importance ───────────────────────────────────────────────────
print("[S10] Plotting feature importance...")
FEAT_MODELS = ["random_forest", "xgboost", "two_stage_xgb_ada", "ordinal_random_forest"]

def _get_best_exp_id(m):
    sub = df[df["model"] == m].sort_values("enhanced_f1_macro", ascending=False)
    return int(sub.iloc[0]["id"]) if not sub.empty else None

def _feat_weights(exp_id, phase="enhanced"):
    row = feat_df[(feat_df["experiment_id"] == exp_id) & (feat_df["phase"] == phase)]
    if row.empty: return []
    wj = row.iloc[0]["weights_json"]
    if not wj: return []
    w = json.loads(wj)
    if isinstance(w, dict) and all(isinstance(v, dict) for v in w.values()):
        w = w.get("stage_1", list(w.values())[0])
    if isinstance(w, dict):
        return sorted(w.items(), key=lambda x: x[1], reverse=True)
    return []

fig, axes = plt.subplots(1, len(FEAT_MODELS), figsize=(18, 6))
for ax, m in zip(axes, FEAT_MODELS):
    exp_id = _get_best_exp_id(m)
    if exp_id is None:
        ax.text(0.5, 0.5, f"No data", ha="center", va="center"); continue
    fw = _feat_weights(exp_id)[:20]
    if not fw:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center"); continue
    features, weights = zip(*fw)
    weights = np.array(weights, dtype=float)
    if weights.max() > 0: weights = weights / weights.max()
    colors = ["#D65F5F" if f in SURVIVAL_FEATURES else "#4878CF" for f in features]
    ax.barh(range(len(features)), weights, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(features))); ax.set_yticklabels(features, fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("Relative Importance", fontsize=8.5)
    strat = df[df["id"]==exp_id]["strategy_label"].values[0]
    f1    = df[df["id"]==exp_id]["enhanced_f1_macro"].values[0]
    ax.set_title(f"{MODEL_DISPLAY.get(m,m)}\n{strat} | F1={f1:.3f}", fontsize=9, fontweight="bold")

patches = [mpatches.Patch(color="#D65F5F", label="Survival features (Cox PH)"),
           mpatches.Patch(color="#4878CF", label="Non-survival features")]
fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5,-0.03))
plt.suptitle("Top-20 Feature Importances (enhanced features, best config per model)",
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_feature_importance.png", bbox_inches="tight")
plt.close()
print("  Saved fig_feature_importance.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FIGURES GENERATED SUCCESSFULLY")
print("="*60)
print(f"\nOutput directory: {OUTPUT_DIR.resolve()}")
import os
for f in sorted(OUTPUT_DIR.glob("*.png")):
    size_kb = os.path.getsize(f) / 1024
    print(f"  {f.name:<45} ({size_kb:>6.1f} KB)")

print(f"\nKey findings:")
print(f"  Best F1  : {df['enhanced_f1_macro'].max():.4f}")
print(f"  Best AUC : {df['enhanced_roc_auc_macro'].max():.4f}")
print(f"  Best model: {df.loc[df['enhanced_f1_macro'].idxmax(),'model_display']}")
print(f"  Best strategy: {df.loc[df['enhanced_f1_macro'].idxmax(),'strategy_label']}")
print()
for fam in ["Base","Ordinal","Two-Stage"]:
    sub = df[df["family"]==fam]
    print(f"  {fam:<12} best F1={sub['enhanced_f1_macro'].max():.4f} "
          f"AUC={sub['enhanced_roc_auc_macro'].max():.4f} "
          f"({sub.loc[sub['enhanced_f1_macro'].idxmax(),'model_display']})")
conn.close()
