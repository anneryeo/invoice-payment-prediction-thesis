# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL STATE
# ══════════════════════════════════════════════════════════════════════════════

# MODELS is populated lazily when Step 4 first renders via the
# "step4-data-loaded" store callback — not at import time.
MODELS: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro"]
METRIC_LABELS = {
    "accuracy":         "Accuracy",
    "precision_macro":  "Precision",
    "recall_macro":     "Recall",
    "f1_macro":         "F1",
    "roc_auc_macro":    "ROC-AUC",
}
CHART_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
PAGE_SIZE = 5  # Default page size of the leaderboard

# ── Display label mappings ────────────────────────────────────────────────────
MODEL_LABELS = {
    "random_forest":        "Random Forest",
    "gaussian_naive_bayes": "Gaussian Naive Bayes",
    "ada_boost":            "AdaBoost",
    "xgboost":              "XGBoost",
    "stacked_ensemble":     "Stacked Ensemble",
    "nn_mlp":               "MLP Neural Net",
    "decision_tree":        "Decision Tree",
    #"logistic_regression":  "Logistic Regression",
    #"svm":                  "SVM",
    "knn":                  "K-Nearest Neighbors",
    #"gradient_boosting":    "Gradient Boosting",
}

# Full strategy name -> short variant label shown in the pill
# All strategies are "SMOTE variants" so we strip the SMOTE prefix where redundant
STRATEGY_LABELS = {
    "none":               "None",
    "smote":              "SMOTE",
    "borderline_smote":   "Borderline",
    "smote_tomek":        "Tomek",
    "smote_enn":          "ENN",
    #"adasyn":             "ADASYN",
    #"random_oversample":  "Random OS",
    #"random_undersample": "Random US",
}

# Reverse of class_mappings.json: int index (as str) -> readable label
_CLASS_MAPPINGS_RAW = {
    "30_days": 0,
    "60_days": 1,
    "90_days": 2,
    "on_time": 3,
}
CLASS_LABELS = {
    str(v): k.replace("_", " ").title()
    for k, v in _CLASS_MAPPINGS_RAW.items()
}


def _model_display(raw_name: str) -> str:
    """Human-readable model name from snake_case."""
    return MODEL_LABELS.get(raw_name.lower(), raw_name.replace("_", " ").title())


def _strategy_display(raw: str) -> str:
    """Short variant label for a balance strategy."""
    return STRATEGY_LABELS.get(raw.lower(), raw.replace("_", " ").title())


def _class_label(cls_key: str) -> str:
    """Readable class label from string index e.g. '0' -> '30 Days'."""
    return CLASS_LABELS.get(str(cls_key), f"Class {cls_key}")


def _params_tooltip(params: dict) -> str:
    """
    Format a hyperparameter dict into a compact multi-line string suitable
    for an HTML title tooltip, e.g.:
        n_estimators: 100
        max_depth: 5
        learning_rate: 0.1
    Returns an empty string when params is empty or not a dict.
    """
    if not isinstance(params, dict) or not params:
        return ""
    return "\n".join(f"{k}: {v}" for k, v in sorted(params.items()))