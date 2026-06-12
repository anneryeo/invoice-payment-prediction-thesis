# machine_learning/utils/io/analysis/registry.py
#
# Single source of truth for model metadata consumed by ResultsAnalyzer,
# visualizations, and the Dash dashboard.
#
# Everything that was a module-level dict in ML_Results_Analysis.py now
# lives here so there is exactly one place to update when a new model
# family or balance strategy is added.

from __future__ import annotations

import seaborn as sns


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL METADATA
# ══════════════════════════════════════════════════════════════════════════════

BASE_MODELS: list[str] = [
    "ada_boost", "decision_tree", "gaussian_naive_bayes",
    "knn", "random_forest", "xgboost",
]

ORDINAL_MODELS: list[str] = [
    "ordinal_ada_boost", "ordinal_random_forest", "ordinal_xgboost",
]

TWO_STAGE_MODELS: list[str] = [
    "two_stage_ada_xgb", "two_stage_rf_ada", "two_stage_rf_rf",
    "two_stage_xgb_ada", "two_stage_xgb_rf", "two_stage_xgb_xgb",
]

ALL_MODELS: list[str] = BASE_MODELS + ORDINAL_MODELS + TWO_STAGE_MODELS

MODEL_DISPLAY: dict[str, str] = {
    "ada_boost":             "AdaBoost",
    "decision_tree":         "Decision Tree",
    "gaussian_naive_bayes":  "Gaussian NB",
    "knn":                   "KNN",
    "random_forest":         "Random Forest",
    "xgboost":               "XGBoost",
    "ordinal_ada_boost":     "Ordinal AdaBoost",
    "ordinal_random_forest": "Ordinal RF",
    "ordinal_xgboost":       "Ordinal XGBoost",
    "two_stage_ada_xgb":     "TS Ada→XGB",
    "two_stage_rf_ada":      "TS RF→Ada",
    "two_stage_rf_rf":       "TS RF→RF",
    "two_stage_xgb_ada":     "TS XGB→Ada",
    "two_stage_xgb_rf":      "TS XGB→RF",
    "two_stage_xgb_xgb":     "TS XGB→XGB",
}

FAMILY_MAP: dict[str, str] = {}
FAMILY_MAP.update({m: "Base"      for m in BASE_MODELS})
FAMILY_MAP.update({m: "Ordinal"   for m in ORDINAL_MODELS})
FAMILY_MAP.update({m: "Two-Stage" for m in TWO_STAGE_MODELS})

ORDINAL_BASE_MAP: dict[str, str] = {
    "ordinal_ada_boost":     "ada_boost",
    "ordinal_random_forest": "random_forest",
    "ordinal_xgboost":       "xgboost",
}

TWO_STAGE_BASE_MAP: dict[str, str] = {
    "two_stage_ada_xgb":  "xgboost",
    "two_stage_rf_ada":   "ada_boost",
    "two_stage_rf_rf":    "random_forest",
    "two_stage_xgb_ada":  "ada_boost",
    "two_stage_xgb_rf":   "random_forest",
    "two_stage_xgb_xgb":  "xgboost",
}


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY METADATA
# ══════════════════════════════════════════════════════════════════════════════

STRATEGY_ORDER: list[str] = [
    "none", "smote", "borderline_smote", "smote_tomek",
    "hybrid@0.5", "hybrid@0.7", "hybrid@0.9",
]

STRATEGY_LABELS: dict[str, str] = {
    "none":             "None",
    "smote":            "SMOTE",
    "borderline_smote": "Borderline SMOTE",
    "smote_tomek":      "SMOTE+Tomek",
    "hybrid@0.5":       "Hybrid @0.5",
    "hybrid@0.7":       "Hybrid @0.7",
    "hybrid@0.9":       "Hybrid @0.9",
}


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS / LABEL METADATA
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES: list[str] = [
    "On-Time (0)", "30-Day (1)", "60-Day (2)", "90-Day (3)",
]


# ══════════════════════════════════════════════════════════════════════════════
#  SURVIVAL FEATURE SETS
# ══════════════════════════════════════════════════════════════════════════════

BASE_SURVIVAL_FEATURES: set[str] = {
    "survival_prob", "hazard", "expected_survival",
    "partial_hazard", "log_partial_hazard",
}

SURVIVAL_FEATURES: set[str] = BASE_SURVIVAL_FEATURES.union(
    {f"surv_prob_{t}" for t in [1, 16, 58, 76, 118, 150, 306, 324]},
    {f"cum_hazard_{t}" for t in [30, 76, 118]},
)


# ══════════════════════════════════════════════════════════════════════════════
#  PALETTES
# ══════════════════════════════════════════════════════════════════════════════

FAMILY_PALETTE: dict[str, str] = {
    "Base":      "#4878CF",
    "Ordinal":   "#6ACC65",
    "Two-Stage": "#D65F5F",
}

STRATEGY_PALETTE: dict[str, str] = dict(
    zip(STRATEGY_ORDER, [str(c) for c in sns.color_palette("tab10", 7)])
)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def strategy_label(balance_strategy: str, undersample_threshold: float | None) -> str:
    """
    Build the human-readable strategy label from raw DB columns.

    >>> strategy_label("hybrid", 0.7)
    'hybrid@0.7'
    >>> strategy_label("smote", None)
    'smote'
    """
    if balance_strategy == "hybrid" and undersample_threshold is not None:
        return f"hybrid@{undersample_threshold:.1f}"
    return balance_strategy


def display_name(model: str) -> str:
    """Return the human-readable display name for a model slug."""
    return MODEL_DISPLAY.get(model, model)


def family(model: str) -> str:
    """Return the family for a model slug ('Base', 'Ordinal', or 'Two-Stage')."""
    return FAMILY_MAP.get(model, "Unknown")
