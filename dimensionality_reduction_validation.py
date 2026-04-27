"""
Comprehensive Validation Testing for LDA/PCA Dimensionality Reduction

This script tests both PCA and LDA with multiple component counts
against a Random Forest classifier, measuring F1-score, precision, recall,
and other metrics across 5-fold cross-validation.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from types import SimpleNamespace

# Ensure repo root is in path
sys.path.insert(0, os.path.abspath("."))

from src.utils.data_loaders.read_settings_json import read_settings_json
from src.modules.feature_engineering.credit_sales_machine_learning import CreditSalesProcessor

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

print("[setup] Loading settings...")
settings = read_settings_json("settings.json")
obs_end = datetime.strptime(settings["Training"]["observation_end"], "%Y/%m/%d")

args = SimpleNamespace(
    observation_end=obs_end,
    target_feature=settings["Training"]["target_feature"],
    test_size=float(settings["Training"]["test_size"]),
    parameters_dir=settings["Training"]["MODEL_PARAMETERS"],
    time_points=[30, 60, 90, 120]
)

print("[data] Loading data...")
df_rev = pd.read_excel("database/revenues_pseudonymized.xlsx")
df_enr = pd.read_excel("database/enrollees_pseudonymized.xlsx")

cs = CreditSalesProcessor(
    df_rev, df_enr, args,
    drop_missing_dtp=True,
    drop_demographic_columns=True,
    drop_helper_columns=True,
    exclude_school_years=[2016, 2017, 2018]
)
df_full = cs.show_data()

# Prepare classifier dataset
df_classifier = df_full[df_full["censor"] == 1].copy().drop(columns=["days_elapsed_until_fully_paid", "censor"])

# Identify feature columns (exclude identity, target, survival features)
drop_cols = {
    'student_id_pseudonimized', 'school_year', 'category_name',
    'dtp_bracket', 'due_date', 'date_fully_paid', 'dtp_bracket',
    'partial_hazard', 'log_partial_hazard', 'expected_survival_time',
    'survival_probability', 'cumulative_hazard'
}
feature_cols = [c for c in df_classifier.columns if c not in drop_cols]

X = df_classifier[feature_cols].fillna(0).astype(np.float32)
y = df_classifier['dtp_bracket'].astype(str)

print(f"\n{'='*80}")
print(f"DATASET SUMMARY")
print(f"{'='*80}")
print(f"Shape: {X.shape}")
print(f"Features: {len(feature_cols)}")
print(f"Classes: {len(y.unique())}")
print(f"Class distribution:\n{y.value_counts()}")

# Standardize features for PCA/LDA
print(f"\n[preprocessing] Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

# ──────────────────────────────────────────────────────────────────────────────
# 1. BASELINE: 50 Full Columns
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"BASELINE: Full {len(feature_cols)} Columns")
print(f"{'='*80}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_scores = {'train_f1': [], 'test_f1': [], 'test_auc': [], 'test_precision': [], 'test_recall': []}

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
    X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
    rf.fit(X_train, y_train)
    
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    y_test_proba = rf.predict_proba(X_test)
    
    baseline_scores['train_f1'].append(f1_score(y_train, y_train_pred, average='macro', zero_division=0))
    baseline_scores['test_f1'].append(f1_score(y_test, y_test_pred, average='macro', zero_division=0))
    baseline_scores['test_precision'].append(precision_score(y_test, y_test_pred, average='macro', zero_division=0))
    baseline_scores['test_recall'].append(recall_score(y_test, y_test_pred, average='macro', zero_division=0))
    
    try:
        baseline_scores['test_auc'].append(roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='macro'))
    except:
        baseline_scores['test_auc'].append(np.nan)
    
    print(f"  Fold {fold_idx+1}: Train F1={baseline_scores['train_f1'][-1]:.4f}, Test F1={baseline_scores['test_f1'][-1]:.4f}")

baseline_f1_mean = np.mean(baseline_scores['test_f1'])
baseline_f1_std = np.std(baseline_scores['test_f1'])

print(f"\n✅ Baseline F1-Macro: {baseline_f1_mean:.4f} (±{baseline_f1_std:.4f})")
print(f"   Baseline Precision: {np.mean(baseline_scores['test_precision']):.4f}")
print(f"   Baseline Recall: {np.mean(baseline_scores['test_recall']):.4f}")
print(f"   Baseline AUC: {np.mean(baseline_scores['test_auc']):.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. VALIDATION FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def validate_dimensionality_reduction(X, y, method='pca', n_components_list=[3, 5, 10, 15, 20]):
    """Test multiple dimensionality reduction approaches."""
    
    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for n_comp in n_components_list:
        print(f"\n  Testing {method.upper()} with {n_comp} components...")
        fold_scores = {
            'train_f1': [], 'test_f1': [], 'test_auc': [], 
            'test_precision': [], 'test_recall': []
        }
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Apply Dimensionality Reduction
            if method == 'pca':
                reducer = PCA(n_components=n_comp, random_state=42)
                X_train_reduced = reducer.fit_transform(X_train)
                X_test_reduced = reducer.transform(X_test)
                variance_explained = reducer.explained_variance_ratio_.sum()
            
            elif method == 'lda':
                max_comp = len(np.unique(y)) - 1
                actual_comp = min(n_comp, max_comp)
                reducer = LDA(n_components=actual_comp)
                X_train_reduced = reducer.fit_transform(X_train, y_train)
                X_test_reduced = reducer.transform(X_test)
                variance_explained = reducer.explained_variance_ratio_.sum()
            
            # Train model
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
            rf.fit(X_train_reduced, y_train)
            
            # Evaluate
            y_train_pred = rf.predict(X_train_reduced)
            y_test_pred = rf.predict(X_test_reduced)
            y_test_proba = rf.predict_proba(X_test_reduced)
            
            fold_scores['train_f1'].append(f1_score(y_train, y_train_pred, average='macro', zero_division=0))
            fold_scores['test_f1'].append(f1_score(y_test, y_test_pred, average='macro', zero_division=0))
            fold_scores['test_precision'].append(precision_score(y_test, y_test_pred, average='macro', zero_division=0))
            fold_scores['test_recall'].append(recall_score(y_test, y_test_pred, average='macro', zero_division=0))
            
            try:
                fold_scores['test_auc'].append(roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='macro'))
            except:
                fold_scores['test_auc'].append(np.nan)
        
        # Average across folds
        results.append({
            'Method': method.upper(),
            'N_Components': n_comp,
            'Variance_Explained_%': round(variance_explained * 100, 2),
            'Train_F1_Mean': round(np.mean(fold_scores['train_f1']), 4),
            'Test_F1_Mean': round(np.mean(fold_scores['test_f1']), 4),
            'Test_F1_Std': round(np.std(fold_scores['test_f1']), 4),
            'Test_Precision': round(np.mean(fold_scores['test_precision']), 4),
            'Test_Recall': round(np.mean(fold_scores['test_recall']), 4),
            'Test_AUC': round(np.mean(fold_scores['test_auc']), 4),
            'Overfitting_Gap': round(np.mean(fold_scores['train_f1']) - np.mean(fold_scores['test_f1']), 4),
        })
        
        print(f"    ✓ F1={results[-1]['Test_F1_Mean']:.4f} (var={results[-1]['Variance_Explained_%']}%)")
    
    return pd.DataFrame(results)

# ──────────────────────────────────────────────────────────────────────────────
# 3. TEST PCA
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"Testing PCA")
print(f"{'='*80}")

pca_results = validate_dimensionality_reduction(
    X_scaled, y, method='pca', 
    n_components_list=[3, 5, 10, 15, 20, 25]
)

print(f"\n{pca_results.to_string(index=False)}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. TEST LDA
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"Testing LDA (max 3 components for 4-class problem)")
print(f"{'='*80}")

lda_results = validate_dimensionality_reduction(
    X_scaled, y, method='lda', 
    n_components_list=[2, 3]
)

print(f"\n{lda_results.to_string(index=False)}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. COMBINE & ANALYZE RESULTS
# ──────────────────────────────────────────────────────────────────────────────

all_results = pd.concat([pca_results, lda_results], ignore_index=True)

print(f"\n{'='*80}")
print(f"SUMMARY: Decision Analysis")
print(f"{'='*80}")
print(f"\nBaseline (50 columns): F1 = {baseline_f1_mean:.4f}")
print(f"Acceptable threshold (5% loss): F1 ≥ {baseline_f1_mean * 0.95:.4f}")
print(f"Unacceptable threshold (>10% loss): F1 < {baseline_f1_mean * 0.90:.4f}")

print(f"\n✅ ACCEPTABLE REDUCTIONS (≥95% baseline F1):")
acceptable = all_results[all_results['Test_F1_Mean'] >= baseline_f1_mean * 0.95].sort_values('Variance_Explained_%', ascending=False)
if len(acceptable) > 0:
    print(acceptable[['Method', 'N_Components', 'Test_F1_Mean', 'Variance_Explained_%', 'Overfitting_Gap']].to_string(index=False))
else:
    print("  None found")

print(f"\n❌ UNACCEPTABLE REDUCTIONS (<90% baseline F1):")
unacceptable = all_results[all_results['Test_F1_Mean'] < baseline_f1_mean * 0.90].sort_values('N_Components')
if len(unacceptable) > 0:
    print(unacceptable[['Method', 'N_Components', 'Test_F1_Mean', 'Variance_Explained_%']].to_string(index=False))
else:
    print("  None found")

# ──────────────────────────────────────────────────────────────────────────────
# 6. VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"Generating visualizations...")
print(f"{'='*80}")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: F1 Score vs Components
ax = axes[0, 0]
for method in ['PCA', 'LDA']:
    data = all_results[all_results['Method'] == method]
    ax.plot(data['N_Components'], data['Test_F1_Mean'], marker='o', label=method, linewidth=2.5, markersize=8)
    ax.fill_between(data['N_Components'], 
                     data['Test_F1_Mean'] - data['Test_F1_Std'],
                     data['Test_F1_Mean'] + data['Test_F1_Std'],
                     alpha=0.2)

ax.axhline(y=baseline_f1_mean, color='red', linestyle='--', label=f'Baseline (50 cols): {baseline_f1_mean:.4f}', linewidth=2.5)
ax.axhline(y=baseline_f1_mean * 0.95, color='orange', linestyle=':', label='95% Threshold', linewidth=2)
ax.axhline(y=baseline_f1_mean * 0.90, color='darkred', linestyle=':', label='90% Threshold', linewidth=2)
ax.set_xlabel('Number of Components', fontsize=11)
ax.set_ylabel('F1-Score (Macro)', fontsize=11)
ax.set_title('F1-Score vs Dimensionality Reduction', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Variance Explained vs Components
ax = axes[0, 1]
for method in ['PCA', 'LDA']:
    data = all_results[all_results['Method'] == method]
    ax.plot(data['N_Components'], data['Variance_Explained_%'], marker='s', label=method, linewidth=2.5, markersize=8)

ax.axhline(y=85, color='green', linestyle=':', label='85% Target', linewidth=2)
ax.set_xlabel('Number of Components', fontsize=11)
ax.set_ylabel('Variance Explained (%)', fontsize=11)
ax.set_title('Variance Explained vs Components', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: Precision vs Recall
ax = axes[1, 0]
for method in ['PCA', 'LDA']:
    data = all_results[all_results['Method'] == method]
    ax.scatter(data['Test_Recall'], data['Test_Precision'], s=150, label=method, alpha=0.7, edgecolors='black', linewidth=1.5)
    for idx, row in data.iterrows():
        ax.annotate(f"{row['N_Components']}", 
                   (row['Test_Recall'], row['Test_Precision']), 
                   fontsize=9, fontweight='bold')

ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Precision vs Recall Trade-off', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: F1 vs Variance (Pareto frontier)
ax = axes[1, 1]
for method in ['PCA', 'LDA']:
    data = all_results[all_results['Method'] == method]
    ax.scatter(data['Variance_Explained_%'], data['Test_F1_Mean'], s=150, label=method, alpha=0.7, edgecolors='black', linewidth=1.5)
    for idx, row in data.iterrows():
        ax.annotate(f"{row['N_Components']}", 
                   (row['Variance_Explained_%'], row['Test_F1_Mean']), 
                   fontsize=9, fontweight='bold')

ax.axhline(y=baseline_f1_mean, color='red', linestyle='--', label='Baseline', linewidth=2.5)
ax.axvline(x=85, color='green', linestyle=':', label='85% Variance Target', linewidth=2)
ax.set_xlabel('Variance Explained (%)', fontsize=11)
ax.set_ylabel('F1-Score', fontsize=11)
ax.set_title('Information Retention vs Performance', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/dimensionality_reduction_validation.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Visualization saved: results/dimensionality_reduction_validation.png")
plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# 7. SAVE RESULTS TO CSV
# ──────────────────────────────────────────────────────────────────────────────

results_csv = 'results/dimensionality_reduction_results.csv'
all_results.to_csv(results_csv, index=False)
print(f"✅ Results saved: {results_csv}")

# Summary statistics
with open('results/dimensionality_reduction_summary.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("DIMENSIONALITY REDUCTION VALIDATION SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Dataset: {X.shape[0]} samples × {X.shape[1]} features\n")
    f.write(f"Target: {len(y.unique())} classes\n")
    f.write(f"CV Strategy: 5-fold Stratified\n\n")
    
    f.write("BASELINE (50 columns):\n")
    f.write(f"  F1-Score: {baseline_f1_mean:.4f} ± {baseline_f1_std:.4f}\n")
    f.write(f"  Precision: {np.mean(baseline_scores['test_precision']):.4f}\n")
    f.write(f"  Recall: {np.mean(baseline_scores['test_recall']):.4f}\n")
    f.write(f"  AUC: {np.mean(baseline_scores['test_auc']):.4f}\n\n")
    
    f.write("ACCEPTABLE REDUCTIONS (≥95% baseline F1):\n")
    acceptable = all_results[all_results['Test_F1_Mean'] >= baseline_f1_mean * 0.95]
    if len(acceptable) > 0:
        f.write(acceptable[['Method', 'N_Components', 'Test_F1_Mean', 'Variance_Explained_%']].to_string(index=False))
    else:
        f.write("  None\n")
    f.write("\n\n")
    
    f.write("RECOMMENDATION:\n")
    best_pca = pca_results[pca_results['Test_F1_Mean'] >= baseline_f1_mean * 0.95].iloc[0] if len(pca_results[pca_results['Test_F1_Mean'] >= baseline_f1_mean * 0.95]) > 0 else None
    best_lda = lda_results[lda_results['Test_F1_Mean'] >= baseline_f1_mean * 0.95].iloc[0] if len(lda_results[lda_results['Test_F1_Mean'] >= baseline_f1_mean * 0.95]) > 0 else None
    
    if best_pca is not None:
        f.write(f"\n✅ PCA with {int(best_pca['N_Components'])} components:\n")
        f.write(f"   - F1: {best_pca['Test_F1_Mean']:.4f} ({(best_pca['Test_F1_Mean']/baseline_f1_mean - 1)*100:+.1f}%)\n")
        f.write(f"   - Variance: {best_pca['Variance_Explained_%']:.1f}%\n")
        f.write(f"   - Reduction: {len(feature_cols)} → {int(best_pca['N_Components'])} features ({(1 - int(best_pca['N_Components'])/len(feature_cols))*100:.1f}% reduction)\n")
    else:
        f.write(f"\n❌ No acceptable PCA reduction found\n")
    
    if best_lda is not None:
        f.write(f"\n✅ LDA with {int(best_lda['N_Components'])} components:\n")
        f.write(f"   - F1: {best_lda['Test_F1_Mean']:.4f} ({(best_lda['Test_F1_Mean']/baseline_f1_mean - 1)*100:+.1f}%)\n")
        f.write(f"   - Variance: {best_lda['Variance_Explained_%']:.1f}%\n")
        f.write(f"   - Reduction: {len(feature_cols)} → {int(best_lda['N_Components'])} features ({(1 - int(best_lda['N_Components'])/len(feature_cols))*100:.1f}% reduction)\n")
    else:
        f.write(f"\n❌ No acceptable LDA reduction found\n")
    
    f.write(f"\n{'='*80}\n")

print(f"✅ Summary saved: results/dimensionality_reduction_summary.txt")

print(f"\n{'='*80}")
print(f"✅ VALIDATION COMPLETE")
print(f"{'='*80}\n")
