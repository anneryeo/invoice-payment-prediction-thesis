"""
test_deployed_model.py
======================
Smoke-test the finalized model pkl against random samples drawn from
df_credit_sales — the same dataframe used to train the model.

Run from the project root:
    python test_deployed_model.py

What is tested
--------------
1. Both pkl files load cleanly.
2. Survival features are generated with the pre-fitted Cox model.
3. Feature matrix is correctly prepared for the pipeline type:
     - TwoStagePipeline  → pass full enhanced matrix (masks applied internally)
     - All other types   → apply pipeline.selector if it exists
4. predict() and predict_proba() run without error.
5. Probability rows sum to 1.0, all values >= 0, no NaNs.
6. A formatted prediction table is printed for manual inspection.
"""

import os
import pickle
import traceback
import pytest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv

from src.utils.data_loaders.read_settings_json import read_settings_json
from src.modules.machine_learning.utils.features.generate_survival_features import generate_survival_features
from src.modules.machine_learning.utils.features.adjust_survival_time_periods import adjust_payment_period


# ─────────────────────────────────────────────────────────────────────────────
# Config — edit these if needed
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_N     = 20    # number of rows to randomly draw
RANDOM_SEED  = 42

# Human-readable names for the encoded class labels.
# The label_encoder in the pkl maps integer indices → these strings.
# Leave empty to use the raw values from the encoder.
CLASS_NAMES_OVERRIDE: dict = {}   # e.g. {0: "on_time", 1: "30d", 2: "60d", 3: "90d"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hdr(title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print('─' * 70)

def _ok(msg):   print(f"  ✓  {msg}")
def _warn(msg): print(f"  ⚠  {msg}")
def _fail(msg): print(f"  ✗  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load settings + pkl files
# ─────────────────────────────────────────────────────────────────────────────

_hdr("1 · Load pkl files")

settings        = read_settings_json()
results_root    = settings["Training"]["RESULTS_ROOT"]
deployed_models = settings["Training"]["DEPLOYED_MODELS"]
if not os.path.isdir(deployed_models):
    pytest.skip("DEPLOYED_MODELS path not found; skipping deployment smoke test.", allow_module_level=True)

# Auto-detect the most recently written finalized_<model>.pkl
_candidates = sorted([
    f for f in os.listdir(deployed_models)
    if f.startswith("finalized_") and f.endswith(".pkl") and "survival" not in f
])
if not _candidates:
    pytest.skip("No finalized model artifacts found in DEPLOYED_MODELS; skipping deployment smoke test.", allow_module_level=True)

model_pkl_path    = os.path.join(deployed_models, _candidates[-1])
survival_pkl_path = os.path.join(deployed_models, "finalized_survival_model.pkl")

print(f"  Model pkl    : {model_pkl_path}")
print(f"  Survival pkl : {survival_pkl_path}")

with open(model_pkl_path,    "rb") as fh: model_bundle    = pickle.load(fh)
with open(survival_pkl_path, "rb") as fh: survival_bundle = pickle.load(fh)

_ok("Both pkl files loaded")

# ── Unpack model bundle ───────────────────────────────────────────────────────
pipeline      = model_bundle["pipeline"]       # fitted BasePipeline subclass
parameters    = model_bundle["parameters"]     # original hyperparameters
features_obj  = model_bundle["features"]       # BasePipeline.features (selected, weights, …)
label_encoder = model_bundle["label_encoder"]  # maps int indices → class labels

# ── Unpack survival bundle ────────────────────────────────────────────────────
cox_model        = survival_bundle["tuner"]
surv_results     = survival_bundle["survival_results"]
best_surv_params = surv_results["best_surv_parameters"]
best_time_points = surv_results["best_time_points"]

pipeline_type = type(pipeline).__name__
model_type    = type(pipeline.model).__name__

_ok(f"Pipeline type    : {pipeline_type}")
_ok(f"Inner model type : {model_type}")
_ok(f"Label classes    : {label_encoder.classes_}")
_ok(f"Selected features: {len(features_obj.selected)}")
_ok(f"Feature method   : {features_obj.method_text}")
_ok(f"Cox C-index      : {surv_results['best_c_index']:.4f}")
_ok(f"Time points      : {best_time_points}")
print(f"\n  Hyperparameters  : {parameters}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load df_credit_sales
# ─────────────────────────────────────────────────────────────────────────────

_hdr("2 · Load df_credit_sales")

cs_path = os.path.join(results_root, "credit_sales_cache.pkl")
if not os.path.exists(cs_path):
    pytest.skip("credit_sales_cache.pkl not found; skipping deployment smoke test.", allow_module_level=True)

with open(cs_path, "rb") as fh:
    df_credit_sales = pickle.load(fh)

_ok(f"df_credit_sales   shape : {df_credit_sales.shape}")
_ok(f"Columns : {list(df_credit_sales.columns)}")

# ── Replicate the exact split from step_3.clean_datasets ─────────────────────
SURVIVAL_COLS     = ["days_elapsed_until_fully_paid", "censor"]
NON_SURVIVAL_COLS = ["due_date", "dtp_bracket"]

df_data      = df_credit_sales[df_credit_sales["censor"] == 1].copy()
df_data.drop(columns=SURVIVAL_COLS, inplace=True)

df_data_surv = df_credit_sales.drop(columns=NON_SURVIVAL_COLS)

_ok(f"df_data (censor==1) : {df_data.shape}")
_ok(f"df_data_surv        : {df_data_surv.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Random sample
# ─────────────────────────────────────────────────────────────────────────────

_hdr(f"3 · Random sample  (n={SAMPLE_N}, seed={RANDOM_SEED})")

rng          = np.random.default_rng(RANDOM_SEED)
sample_pos   = rng.choice(len(df_data), size=min(SAMPLE_N, len(df_data)), replace=False)
sample_pos   = np.sort(sample_pos)           # keep rows in original order

X_sample_raw = df_data.iloc[sample_pos].copy().reset_index(drop=True)

_ok(f"Sample shape : {X_sample_raw.shape}")
print()
print(X_sample_raw.to_string(max_cols=10, show_dimensions=True))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Generate survival features for the sample
#
#     generate_survival_features needs the *full* survival dataframe (X_surv,
#     T, E) to fit/apply the hazard model, but only needs the sample rows as
#     the X_train argument (the rows we want to transform).
# ─────────────────────────────────────────────────────────────────────────────

_hdr("4 · Generate survival features")

X_surv_full = df_data_surv.drop(columns=["days_elapsed_until_fully_paid", "censor"])
T_full      = adjust_payment_period(df_data_surv["days_elapsed_until_fully_paid"])
E_full      = df_data_surv["censor"]

# Pass fitted_cph so Cox is NOT refitted — just used for inference
X_enhanced = generate_survival_features(
    X_surv=X_surv_full,
    T=T_full,
    E=E_full,
    X_train=X_sample_raw,   # only the sample rows get transformed
    X_test=None,
    best_params=best_surv_params,
    time_points=best_time_points,
    fitted_cph=cox_model,
)

_ok(f"X_enhanced shape : {X_enhanced.shape}")

# Convert to numpy — all pipelines were fitted on numpy arrays via DataPreparer
X_enhanced_np = (
    X_enhanced.values if isinstance(X_enhanced, pd.DataFrame) else np.asarray(X_enhanced)
)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Prepare the feature matrix for prediction
#
#     TwoStagePipeline: pass the full enhanced matrix — mask1_/mask2_ are
#       stored on TwoStageClassifier and applied internally in predict_proba.
#
#     All other pipelines (Ordinal, XGBoost, etc.): if feature selection ran,
#       pipeline.selector is a fitted SelectFromModel. Apply its transform so
#       the model receives only the columns it was trained on.
# ─────────────────────────────────────────────────────────────────────────────

_hdr("5 · Prepare feature matrix for prediction")

is_two_stage = "TwoStage" in pipeline_type or "TwoStage" in model_type

if is_two_stage:
    # Masks are applied internally inside TwoStageClassifier.predict_proba
    X_predict = X_enhanced_np
    _ok(f"TwoStagePipeline detected — passing full matrix ({X_predict.shape[1]} cols)")
elif hasattr(pipeline, "selector") and pipeline.selector is not None:
    X_predict = pipeline.selector.transform(X_enhanced_np)
    _ok(f"Selector applied: {X_enhanced_np.shape[1]} → {X_predict.shape[1]} features")
else:
    X_predict = X_enhanced_np
    _ok(f"No selector — using full enhanced matrix ({X_predict.shape[1]} cols)")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Predict
# ─────────────────────────────────────────────────────────────────────────────

_hdr("6 · Predict")

raw_preds   = None
class_preds = None
probas      = None

try:
    raw_preds   = pipeline.model.predict(X_predict)
    class_preds = label_encoder.inverse_transform(raw_preds)
    _ok(f"predict()       → {raw_preds.shape}  unique: {np.unique(raw_preds)}")
    _ok(f"decoded labels  → unique: {np.unique(class_preds)}")
except Exception:
    _fail("predict() failed:")
    traceback.print_exc()

try:
    probas = pipeline.model.predict_proba(X_predict)
    _ok(f"predict_proba() → {probas.shape}")
except Exception:
    _fail("predict_proba() failed:")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

_hdr("7 · Sanity checks")

if probas is not None:
    # NaNs
    nan_count = np.isnan(probas).sum()
    if nan_count == 0:
        _ok("No NaN values in probabilities")
    else:
        _fail(f"{nan_count} NaN values in probabilities")

    # Non-negative
    neg_count = (probas < 0).sum()
    if neg_count == 0:
        _ok("All probabilities >= 0")
    else:
        _fail(f"{neg_count} negative probability values")

    # Row sums
    row_sums      = probas.sum(axis=1)
    max_deviation = float(np.abs(row_sums - 1.0).max())
    if max_deviation < 1e-5:
        _ok(f"Row sums ≈ 1.0  (max deviation {max_deviation:.2e})")
    else:
        _warn(f"Row sums deviate from 1.0 by up to {max_deviation:.6f}")

    # Per-class mean probability
    class_labels_all = (
        CLASS_NAMES_OVERRIDE if CLASS_NAMES_OVERRIDE
        else {i: str(c) for i, c in enumerate(label_encoder.classes_)}
    )
    print()
    print("  Per-class mean probability across sample:")
    for k in range(probas.shape[1]):
        label    = class_labels_all.get(k, str(k))
        col_mean = probas[:, k].mean()
        bar      = "█" * int(col_mean * 40)
        print(f"    class {k} ({label:>10})  {col_mean:.4f}  {bar}")

if raw_preds is not None:
    print()
    print("  Prediction distribution:")
    for cls, cnt in zip(*np.unique(raw_preds, return_counts=True)):
        label = class_labels_all.get(int(cls), str(cls))
        print(f"    class {cls} ({label:>10}) : {cnt:3d} / {len(raw_preds)}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Survival model smoke-test
#     Refit StandardScaler on the full df_data_surv (same as step_5) and
#     score the pre-fitted Cox model to confirm it still predicts correctly.
# ─────────────────────────────────────────────────────────────────────────────

_hdr("8 · Survival model smoke-test")

try:
    X_raw = df_data_surv.drop(
        columns=["days_elapsed_until_fully_paid", "censor"]
    ).astype(float)
    scaler   = StandardScaler()
    X_scaled = np.clip(scaler.fit_transform(X_raw), -10, 10)

    T = adjust_payment_period(df_data_surv["days_elapsed_until_fully_paid"])
    E = df_data_surv["censor"]
    y_surv = Surv.from_arrays(
        event=E.astype(bool).values,
        time=T.astype(float).values,
    )
    c_index = float(cox_model.score(X_scaled, y_surv))
    delta   = abs(c_index - surv_results["best_c_index"])
    _ok(f"Cox C-index (full data)  : {c_index:.4f}")
    _ok(f"Cox C-index (saved)      : {surv_results['best_c_index']:.4f}")
    if delta > 0.02:
        _warn(f"C-index differs by {delta:.4f} — data may have changed since training")
    else:
        _ok(f"C-index delta            : {delta:.4f}  (within tolerance)")
except Exception:
    _fail("Survival smoke-test failed:")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Prediction table
# ─────────────────────────────────────────────────────────────────────────────

_hdr("9 · Prediction table")

if probas is not None and class_preds is not None:
    class_labels_all = (
        CLASS_NAMES_OVERRIDE if CLASS_NAMES_OVERRIDE
        else {i: str(c) for i, c in enumerate(label_encoder.classes_)}
    )
    proba_cols = {
        f"P({class_labels_all.get(k, k)})": probas[:, k].round(4)
        for k in range(probas.shape[1])
    }
    result_df = pd.DataFrame({
        "sample_row": sample_pos,
        "predicted":  class_preds,
        **proba_cols,
    })
    print()
    print(result_df.to_string(index=False))
else:
    _warn("Prediction table skipped — earlier step failed.")

print()
_hdr("Done")
print()

