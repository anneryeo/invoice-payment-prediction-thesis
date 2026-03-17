# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL STATE
# ══════════════════════════════════════════════════════════════════════════════

# MODELS and CLASS_LABELS are both populated lazily when Step 4 first renders
# via the "step4-data-loaded" store callback — not at import time.
MODELS: dict = {}

# Maps string class index → human-readable label.
# e.g. {"0": "30 Days", "1": "60 Days", "2": "90 Days", "3": "On Time"}
# Populated at runtime from session["class_mappings"] via set_class_labels().
CLASS_LABELS: dict[str, str] = {}


def set_class_labels(class_mappings: dict) -> None:
    """
    Populate CLASS_LABELS from the class_mappings dict returned by SessionStore.

    ``class_mappings`` is the raw label→int mapping stored by the training
    pipeline, e.g. {"30_days": 0, "60_days": 1, "90_days": 2, "on_time": 3}.
    This function inverts it to int-index (as str) → readable label and writes
    the result into the module-level CLASS_LABELS dict in-place so that every
    module that imported CLASS_LABELS sees the updated values immediately.

    Call this once, right after MODELS is populated, inside the
    "step4-data-loaded" callback.
    """
    CLASS_LABELS.clear()
    for raw_label, int_index in class_mappings.items():
        readable = raw_label.replace("_", " ").title()
        CLASS_LABELS[str(int_index)] = readable


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
    "random_forest":         "Random Forest",
    "gaussian_naive_bayes":  "Gaussian Naive Bayes",
    "ada_boost":             "AdaBoost",
    "xgboost":               "XGBoost",
    "decision_tree":         "Decision Tree",
    #"logistic_regression":   "Logistic Regression",
    #"svm":                   "SVM",
    "knn":                   "K-Nearest Neighbors",
    #"gradient_boosting":     "Gradient Boosting",
    "nn_mlp":                "MLP Neural Net",
    "ordinal_xgboost":       "Ordinal - XGBoost",
    "ordinal_random_forest": "Ordinal - Random Forest",
    "ordinal_ada_boost":     "Ordinal - AdaBoost",
    "two_stage_xgb_xgb":     "Two Stage - (XGB -> XGB)",
    "two_stage_xgb_rf":      "Two Stage - (XGB -> RF)",
    "two_stage_rf_rf":       "Two Stage - (RF -> RF)",
    "two_stage_xgb_ada":     "Two Stage - (XGB -> Ada)",
    "two_stage_rf_ada":      "Two Stage - (RF -> Ada)",
    "two_stage_ada_xgb":     "Two Stage - (Ada -> XGB)",
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


def _model_display(raw_name: str) -> str:
    """Human-readable model name from snake_case."""
    return MODEL_LABELS.get(raw_name.lower(), raw_name.replace("_", " ").title())


def _strategy_display(raw: str) -> str:
    """Short variant label for a balance strategy."""
    return STRATEGY_LABELS.get(raw.lower(), raw.replace("_", " ").title())


def _class_label(cls_key: str) -> str:
    """
    Readable class label from a string index, e.g. '0' -> '30 Days'.

    Looks up the runtime CLASS_LABELS dict (populated from the session's
    class_mappings).  Falls back to a generic 'Class N' string when the key
    is not present so charts never break during a cold load.
    """
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
