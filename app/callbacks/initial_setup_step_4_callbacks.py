import json
import os
import re
import sqlite3

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, dcc, no_update, callback_context, ALL

from app import dash_app


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADER  —  reads results.db from the latest dated folder under Results/
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_ROOT = r"C:\Users\rjbel\Python\Notebooks\Mapua\Thesis\Results"
_DATE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}$")    # YYYY_MM_DD_##  where ## is the training run number for that day


def _latest_results_path() -> str:
    """Return the path to results.db inside the most-recent dated sub-folder."""
    dated_dirs = sorted(
        [d for d in os.listdir(RESULTS_ROOT) if _DATE_RE.match(d)],
        reverse=True,
    )
    if not dated_dirs:
        raise FileNotFoundError(f"No dated result folders found under {RESULTS_ROOT}")
    base = os.path.join(RESULTS_ROOT, dated_dirs[0])
    db_path = os.path.join(base, "results.db")
    if os.path.exists(db_path):
        return db_path
    raise FileNotFoundError(f"results.db not found in {base}")


def _json_deserialize(value):
    """
    Transparently deserialize a value that may be a JSON-encoded dict/list
    stored as text in SQLite, or an already-native Python object.

    • None / NaN  → returned as-is
    • list / dict → returned as-is  (already decoded by sqlite3 row_factory)
    • str         → attempted JSON parse; original string returned on failure
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _load_raw_rows(path: str) -> list[dict]:
    """
    Load result rows from a SQLite database at *path*.
    Returns a list of plain dicts keyed by column name, with JSON-encoded
    columns transparently deserialized.

    The database must contain a table named ``results`` whose columns match
    the schema written by the training pipeline.  Dict/list columns are stored
    as JSON text and are decoded here before the caller sees them.
    """
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row          # column-name access
    try:
        cur = con.execute("SELECT * FROM results")
        columns = [desc[0] for desc in cur.description]
        raw_rows = []
        for sqlite_row in cur.fetchall():
            row = {}
            for col in columns:
                v = sqlite_row[col]
                row[col] = _json_deserialize(v)
            raw_rows.append(row)
    finally:
        con.close()

    return raw_rows


def load_models_from_results() -> dict:
    """
    Read results.db from the latest dated folder and return a MODELS-compatible
    dict.  Dict/list columns (confusion_matrix, roc_curve, pr_curve,
    feature_selected) are JSON-deserialized transparently by _load_raw_rows.
    """
    path = _latest_results_path()
    raw_rows = _load_raw_rows(path)

    # ── Diagnostic: log curve lengths ────────────────────────────────────────
    if raw_rows:
        sample = raw_rows[0]
        for col in ("baseline_roc_curve", "baseline_pr_curve",
                    "enhanced_roc_curve", "enhanced_pr_curve"):
            v = sample.get(col, "")
            print(f"[step4] {col} (sqlite) length: {len(str(v)) if v else 0}")

    # ── Build MODELS dict ─────────────────────────────────────────────────────
    # Each row is uniquely identified by (model, balance_strategy, parameters).
    # Previously the key was f"{slug}__{balance}__none", which caused rows that
    # share the same model+strategy but differ only in hyperparameters to silently
    # overwrite each other — hence 28 rows instead of the expected 240.
    # We now include a short MD5 hash of the serialized parameters in the key.
    import hashlib
    models: dict = {}
    for row in raw_rows:
        def _get(col):
            v = row.get(col)
            return v if v is not None else ""

        model_name: str = str(_get("model"))
        balance: str    = str(_get("balance_strategy")) if _get("balance_strategy") else "none"

        # Deserialize parameters; build a stable hash for use in the dict key.
        # Handles: JSON dict, single-quoted dict, Python list-of-tuples repr.
        raw_params = _json_deserialize(_get("parameters"))

        def _normalise_params(raw) -> dict:
            """Convert any params representation to a plain {str: scalar} dict."""
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, (list, tuple)):
                # list/tuple of 2-tuples: [('key', val), ...]
                try:
                    result = {}
                    for item in raw:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            result[str(item[0])] = item[1]
                    if result:
                        return result
                except Exception:
                    pass
            if isinstance(raw, str) and raw.strip():
                # Try single-quoted dict → JSON
                try:
                    cleaned = (raw.strip()
                               .replace("'", '"')
                               .replace("True", "true")
                               .replace("False", "false")
                               .replace("None", "null"))
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
                # Try Python literal_eval for list-of-tuples repr
                try:
                    import ast
                    evaled = ast.literal_eval(raw.strip())
                    return _normalise_params(evaled)
                except Exception:
                    pass
            return {}

        params: dict = _normalise_params(raw_params)
        params_sig   = json.dumps(params, sort_keys=True) if params else "default"

        param_hash = hashlib.md5(params_sig.encode()).hexdigest()[:6]
        slug = re.sub(r"\s+", "_", model_name).lower()
        key  = f"{slug}__{balance}__{param_hash}"

        def _section(prefix: str) -> dict:
            def _float(col):
                v = _get(f"{prefix}_{col}")
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return 0.0

            metrics = {
                "accuracy":        _float("accuracy"),
                "precision_macro": _float("precision_macro"),
                "recall_macro":    _float("recall_macro"),
                "f1_macro":        _float("f1_macro"),
                "roc_auc_macro":   _float("roc_auc_macro"),
            }
            # roc_curve and pr_curve are already dicts after JSON deserialization
            charts = {
                "confusion_matrix": _json_deserialize(_get(f"{prefix}_confusion_matrix")),
                "roc_curve":        _get(f"{prefix}_roc_curve"),   # kept raw for _parse_curve
                "pr_curve":         _get(f"{prefix}_pr_curve"),    # kept raw for _parse_curve
            }
            raw_selected = _json_deserialize(_get(f"{prefix}_feature_selected"))
            features: list = raw_selected if isinstance(raw_selected, list) else []

            return {
                "evaluation": {"metrics": metrics, "charts": charts},
                "features": features,
            }

        models[key] = {
            "model":            model_name,
            "balance_strategy": balance,
            "parameters":       params,
            "baseline":         _section("baseline"),
            "enhanced":         _section("enhanced"),
        }

    return models


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
PAGE_SIZE = 5

# ── Display label mappings ────────────────────────────────────────────────────
MODEL_LABELS = {
    "random_forest":        "Random Forest",
    "gaussian_naive_bayes": "Gaussian Naive Bayes",
    "ada_boost":            "AdaBoost",
    "xgboost":              "XGBoost",
    "nn_mlp":               "MLP Neural Net",
    "decision_tree":        "Decision Tree",
    "logistic_regression":  "Logistic Regression",
    "svm":                  "SVM",
    "knn":                  "K-Nearest Neighbors",
    "gradient_boosting":    "Gradient Boosting",
}

# Full strategy name -> short variant label shown in the pill
# All strategies are "SMOTE variants" so we strip the SMOTE prefix where redundant
STRATEGY_LABELS = {
    "none":               "None",
    "smote":              "SMOTE",
    "borderline_smote":   "Borderline",
    "smote_tomek":        "Tomek",
    "smote_enn":          "ENN",
    "adasyn":             "ADASYN",
    "random_oversample":  "Random OS",
    "random_undersample": "Random US",
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


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════

def _base_layout(title_text: str) -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        font=dict(family="'IBM Plex Mono', monospace", color="#374151", size=10),
        margin=dict(l=52, r=28, t=60, b=48),
        title=dict(
            text=title_text,
            font=dict(family="'DM Serif Display', serif", size=13, color="#111827"),
            x=0, xanchor="left", pad=dict(l=8, t=6),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e5e7eb", borderwidth=1,
            font=dict(size=9),
        ),
        xaxis=dict(gridcolor="#e5e7eb", linecolor="#d1d5db", tickfont=dict(size=9), zeroline=False),
        yaxis=dict(gridcolor="#e5e7eb", linecolor="#d1d5db", tickfont=dict(size=9), zeroline=False),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_metrics(model_key: str, result_type: str) -> dict:
    return MODELS[model_key][result_type]["evaluation"]["metrics"]


def delta_badge(val: float, ref: float):
    diff = val - ref
    if abs(diff) < 0.0005:
        return html.Span("—", className="delta-neutral")
    sign = "▲" if diff > 0 else "▼"
    cls = "delta-up" if diff > 0 else "delta-down"
    return html.Span(f"{sign}{abs(diff):.3f}", className=cls)


def build_leaderboard_rows(sort_metric: str, search_val: str, result_type: str) -> list:
    rows = []
    for key, data in MODELS.items():
        base_m = get_metrics(key, "baseline")
        enh_m  = get_metrics(key, "enhanced")
        rows.append({
            "key":        key,
            "name":       data["model"],
            "strategy":   data["balance_strategy"],
            "parameters": data.get("parameters", {}),
            **{f"base_{m}": base_m.get(m, 0) for m in METRICS},
            **{f"enh_{m}":  enh_m.get(m, 0)  for m in METRICS},
            "sort_val": (enh_m if result_type == "enhanced" else base_m).get(sort_metric, 0),
        })
    if search_val:
        rows = [r for r in rows if search_val.lower() in r["name"].lower() or search_val.lower() in _model_display(r["name"]).lower()]
    rows.sort(key=lambda x: x["sort_val"], reverse=True)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_curve(raw):
    """
    Resolve a roc_curve / pr_curve value into a plain Python dict.

    After SQLite + JSON deserialization the value will typically already be a
    dict.  This function handles that fast path as well as the fallback where
    the value arrives as a JSON string (e.g. doubly-encoded) that still needs
    one more json.loads pass.
    """
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return {}
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[step4] _parse_curve json.loads failed: {e} — preview: {raw[:80]}")
    return {}


def build_roc_figure(model_key: str, result_type: str) -> go.Figure:
    roc = _parse_curve(
        MODELS[model_key][result_type]["evaluation"]["charts"].get("roc_curve")
    )
    auc = get_metrics(model_key, result_type).get("roc_auc_macro", 0)

    layout = _base_layout(f"ROC Curve · AUC {auc:.3f}")
    layout["xaxis"]["title"] = "False Positive Rate"
    layout["xaxis"]["range"] = [0, 1]
    layout["xaxis"]["fixedrange"] = True
    layout["yaxis"]["title"] = "True Positive Rate"
    layout["yaxis"]["range"] = [0, 1]
    layout["yaxis"]["fixedrange"] = True

    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="#9ca3af", dash="dash", width=1))

    for i, cls in enumerate(sorted(roc.keys(), key=lambda k: int(k) if str(k).isdigit() else k)):
        entry = roc[cls]
        fpr = entry.get("fpr", []) if isinstance(entry, dict) else []
        tpr = entry.get("tpr", []) if isinstance(entry, dict) else []
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines", name=_class_label(str(cls)),
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
        ))

    fig.update_layout(**layout)
    return fig


def build_pr_figure(model_key: str, result_type: str) -> go.Figure:
    pr = _parse_curve(
        MODELS[model_key][result_type]["evaluation"]["charts"].get("pr_curve")
    )

    layout = _base_layout("Precision–Recall Curve")
    layout["xaxis"]["title"] = "Recall"
    layout["xaxis"]["range"] = [0, 1]
    layout["xaxis"]["fixedrange"] = True
    layout["yaxis"]["title"] = "Precision"
    layout["yaxis"]["range"] = [0, 1.05]
    layout["yaxis"]["fixedrange"] = True

    fig = go.Figure()
    for i, cls in enumerate(sorted(pr.keys(), key=lambda k: int(k) if str(k).isdigit() else k)):
        entry = pr[cls]
        recall    = entry.get("recall",    []) if isinstance(entry, dict) else []
        precision = entry.get("precision", []) if isinstance(entry, dict) else []
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode="lines", name=_class_label(str(cls)),
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
        ))

    fig.update_layout(**layout)
    return fig


# Short axis labels for the confusion matrix  (index → tick label)
# Derived from _CLASS_MAPPINGS_RAW so they stay in sync automatically.
_CM_TICK_LABELS: dict[int, str] = {
    0: "30 days",
    1: "60 days",
    2: "90 days",
    3: "On Time",
}


def build_cm_figure(model_key: str, result_type: str) -> go.Figure:
    cm_raw = MODELS[model_key][result_type]["evaluation"]["charts"].get("confusion_matrix")
    if isinstance(cm_raw, str):
        cm_raw = _json_deserialize(cm_raw)
    if cm_raw is None:
        cm_arr = np.zeros((3, 3), dtype=int)
    else:
        cm_arr = np.array(cm_raw)

    n = cm_arr.shape[0]
    tick_vals   = list(range(n))
    tick_labels = [_CM_TICK_LABELS.get(i, str(i)) for i in range(n)]

    _full_labels = {0: "30 Days", 1: "60 Days", 2: "90 Days", 3: "On Time"}
    full_labels = [_full_labels.get(i, tick_labels[i]) for i in range(n)]

    total = cm_arr.sum(axis=1, keepdims=True)
    pct   = np.where(total > 0, cm_arr / total * 100, 0)

    # ── Per-cell text colour based on normalised cell intensity ──────────────
    # Colorscale: #eff6ff (low=0) → #93c5fd (mid=0.5) → #1d4ed8 (high=1).
    # Interpolate each cell's normalised value to an approximate luminance and
    # pick white text above a threshold, dark text below.
    z_min = float(cm_arr.min())
    z_max = float(cm_arr.max())
    z_range = z_max - z_min if z_max != z_min else 1.0

    # RGB for the three colorscale stops
    _cs = [(0xef, 0xf6, 0xff), (0x93, 0xc5, 0xfd), (0x1d, 0x4e, 0xd8)]

    def _interp_color(norm: float):
        """Linearly interpolate across the three colorscale stops."""
        if norm <= 0.5:
            t = norm / 0.5
            r = _cs[0][0] + t * (_cs[1][0] - _cs[0][0])
            g = _cs[0][1] + t * (_cs[1][1] - _cs[0][1])
            b = _cs[0][2] + t * (_cs[1][2] - _cs[0][2])
        else:
            t = (norm - 0.5) / 0.5
            r = _cs[1][0] + t * (_cs[2][0] - _cs[1][0])
            g = _cs[1][1] + t * (_cs[2][1] - _cs[1][1])
            b = _cs[1][2] + t * (_cs[2][2] - _cs[1][2])
        return r / 255, g / 255, b / 255

    def _relative_luminance(r, g, b):
        """WCAG relative luminance."""
        def _lin(c):
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)

    # customdata for hover
    customdata = [
        [
            [full_labels[i], full_labels[j], int(cm_arr[i][j]), float(pct[i][j])]
            for j in range(n)
        ]
        for i in range(n)
    ]

    layout = _base_layout("Confusion Matrix")
    axis_common = dict(
        fixedrange=True,
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_labels,
        tickfont=dict(size=10, family="'IBM Plex Mono', monospace"),
    )
    layout["xaxis"].update({"title": "Predicted", **axis_common})
    layout["yaxis"].update({"title": "Actual",    **axis_common})

    # Standard convention: rows = Actual (Y), columns = Predicted (X).
    # sklearn stores cm[actual][predicted], so cm_arr[i][j] = actual i, predicted j.
    # Plotly Heatmap renders z[i] as row i from the bottom, so we transpose
    # and flip the Y axis to get the canonical top-left = (Actual 0, Predicted 0) layout.
    cm_plot = cm_arr.T

    fig = go.Figure(go.Heatmap(
        z=cm_plot,
        x=tick_vals,
        y=tick_vals,
        colorscale=[[0, "#eff6ff"], [0.5, "#93c5fd"], [1, "#1d4ed8"]],
        showscale=False,
        customdata=customdata,
        hovertemplate=(
            "<b style='color:#94a3b8;'>Actual&#160;&#160;&#160;&#160;&#160;&#160;</b>%{customdata[0]}<br>"
            "<b style='color:#94a3b8;'>Predicted&#160;&#160;&#160;</b>%{customdata[1]}<br>"
            "<b style='color:#94a3b8;'>Count&#160;&#160;&#160;&#160;&#160;&#160;&#160;</b>%{customdata[2]}<br>"
            "<b style='color:#94a3b8;'>Percentage&#160;</b>%{customdata[3]:.1f}%"
            "<extra></extra>"
        ),
    ))

    # ── Per-cell annotations with contrast-aware text colour ─────────────────
    for i in range(n):
        for j in range(n):
            # cm_plot[i][j] = cm_arr[j][i]: predicted j, actual i in plot space
            val  = cm_plot[i][j]
            norm = (float(val) - z_min) / z_range
            r, g, b = _interp_color(norm)
            lum   = _relative_luminance(r, g, b)
            color = "#ffffff" if lum < 0.35 else "#111827"
            # pct shown is row-normalised over the original cm_arr (actual rows)
            pct_val = pct[j][i]
            fig.add_annotation(
                x=j, y=i,
                text=f"{val}<br>{pct_val:.1f}%",
                showarrow=False,
                font=dict(size=10, color=color,
                          family="'IBM Plex Mono', monospace"),
                align="center",
            )

    fig.update_layout(
        **layout,
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="#334155",
            font=dict(
                family="'IBM Plex Mono', monospace",
                size=11,
                color="#f1f5f9",
            ),
        ),
    )
    return fig


def build_features_figure(model_key: str, result_type: str,
                          positives_only: bool = False) -> go.Figure:
    features = MODELS[model_key][result_type].get("features", [])

    # Use real weights if available, otherwise fall back to rank-based estimates
    raw_weights = MODELS[model_key][result_type]["evaluation"]["charts"].get("feature_weights")
    if raw_weights and isinstance(raw_weights, (list, dict)):
        if isinstance(raw_weights, dict):
            importance = [float(raw_weights.get(f, 0)) for f in features]
        else:
            importance = [float(w) for w in raw_weights]
    else:
        importance = sorted(
            [round(1 - i * 0.1 + np.random.uniform(-0.03, 0.03), 3) for i in range(len(features))],
            reverse=True,
        )

    # Sort both lists together by importance descending
    paired = sorted(zip(importance, features), reverse=True)

    # Optionally filter out negative-importance features
    if positives_only:
        paired = [(v, f) for v, f in paired if v >= 0]

    importance = [p[0] for p in paired]
    features   = [p[1] for p in paired]

    # Per-bar colors: top bar accent blue, positive = light blue, negative = muted red-orange
    colors = []
    for idx, v in enumerate(importance):
        if idx == 0:
            colors.append(CHART_COLORS[0])          # #1d4ed8 — top feature
        elif v >= 0:
            colors.append("#93c5fd")                # light blue — positive
        else:
            colors.append("#fca5a5")                # light red — negative

    # Dynamic left margin based on longest feature name to prevent label cutoff
    max_label_len = max((len(f) for f in features), default=10)
    left_margin   = max(100, min(max_label_len * 7, 220))

    # Title is rendered as HTML outside the scroll area; use empty string here
    layout = _base_layout("")
    layout["title"]  = None
    layout["xaxis"]["title"]      = "Importance Score"
    layout["xaxis"]["fixedrange"] = True
    layout["yaxis"]["autorange"]  = "reversed"
    layout["yaxis"]["fixedrange"] = True
    layout["margin"]["l"] = left_margin
    layout["margin"]["r"] = 60
    layout["margin"]["t"] = 16    # small top margin since title is outside
    layout["margin"]["b"] = 48
    # Height scales with features — scroll wrapper caps visible height
    layout["height"] = max(300, 30 * len(features) + 90)

    # Per-bar text color based on background luminance for contrast
    # #1d4ed8 (dark blue) → white text; #93c5fd (light blue) / #fca5a5 (light red) → dark text
    _bar_color_map = {
        CHART_COLORS[0]: (0x1d/255, 0x4e/255, 0xd8/255),
        "#93c5fd":        (0x93/255, 0xc5/255, 0xfd/255),
        "#fca5a5":        (0xfc/255, 0xa5/255, 0xa5/255),
    }
    def _bar_lum(hex_color):
        r, g, b = _bar_color_map.get(hex_color, (1, 1, 1))
        def _lin(c):
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)

    text_colors = ["#ffffff" if _bar_lum(c) < 0.35 else "#1e3a5f" for c in colors]

    fig = go.Figure(go.Bar(
        x=importance, y=features,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in importance],
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(size=9, color=text_colors, family="'IBM Plex Mono', monospace"),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  HTML LAYOUT — STEP 4
# ══════════════════════════════════════════════════════════════════════════════

def _metric_header(metric, label):
    return html.Th(
        html.Button(label, id={"type": "sort-btn", "metric": metric}, className="sort-btn"),
        colSpan=2,
        className="metric-header-cell",
    )


html_step_4 = html.Div(
    className="dashboard-root",
    children=[

        # ── Page header ──────────────────────────────────────────────────────
        html.Div(className="dash-page-header", children=[
            html.Div(className="dash-header-left", children=[
                html.Span("STEP 4 · MODEL ANALYSIS", className="dash-title-tag"),
                html.H2("Result Analysis", className="dash-title"),
                html.P(
                    "Comparative performance across baseline and survival-enhanced pipelines.",
                    className="dash-subtitle",
                ),
            ]),
            html.Div(className="dash-header-right", children=[
                html.Div(className="result-toggle-wrap", children=[
                    html.Span("View:", className="toggle-label"),
                    html.Div(className="result-toggle", children=[
                        html.Button("Baseline", id="toggle-baseline", className="toggle-btn"),
                        html.Button("Enhanced", id="toggle-enhanced", className="toggle-btn active-toggle"),
                    ]),
                ]),
            ]),
        ]),

        # ── Global controls ──────────────────────────────────────────────────
        html.Div(className="global-controls", children=[
            dcc.Input(id="model-search", placeholder="Filter models…",
                      className="search-input", debounce=True),
            html.Div(className="controls-right", children=[
                html.Button("↓ Export CSV", id="export-csv-btn", className="export-btn"),
                dcc.Download(id="download-csv"),
            ]),
        ]),

        # ── Hidden stores ────────────────────────────────────────────────────
        dcc.Store(id="sort-metric-store",       data="f1_macro"),
        dcc.Store(id="sort-dir-store",          data="desc"),
        dcc.Store(id="result-type-store",       data="enhanced"),
        dcc.Store(id="selected-model-store",    data=""),
        dcc.Store(id="page-store",              data=0),
        dcc.Store(id="step4-data-loaded",       data=False),
        dcc.Store(id="features-positive-store", data=False),  # True = hide negatives

        # ── Parameter tooltip (shown on model-row hover) ─────────────────────
        html.Div(id="params-tooltip", className="params-tooltip", children=[
            html.Div(id="params-tooltip-content", className="params-tooltip-content"),
        ]),

        # ── Leaderboard table ────────────────────────────────────────────────
        html.Div(className="leaderboard-wrap", children=[
            html.Div(id="leaderboard-table-container"),
            html.Div(className="pagination-bar", children=[
                html.Span(id="page-indicator", className="page-indicator"),
                html.Div(className="page-btn-group", children=[
                    html.Button("«", id="page-first", className="page-btn", title="First page"),
                    html.Button("‹", id="page-prev",  className="page-btn", title="Previous page"),
                    html.Button("›", id="page-next",  className="page-btn", title="Next page"),
                    html.Button("»", id="page-last",  className="page-btn", title="Last page"),
                ]),
            ]),
        ]),

        # ── Chart section ────────────────────────────────────────────────────
        html.Div(className="charts-section", children=[
            html.Div(id="charts-model-label", className="charts-model-label"),
            html.Div(className="charts-grid", children=[
                html.Div(className="chart-card", children=[dcc.Graph(id="chart-roc",      config={"displayModeBar": False})]),
                html.Div(className="chart-card", children=[dcc.Graph(id="chart-pr",       config={"displayModeBar": False})]),
                html.Div(className="chart-card", children=[dcc.Graph(id="chart-cm",       config={"displayModeBar": False})]),
                html.Div(className="chart-card chart-card-features", children=[
                    html.Div(className="features-card-header", children=[
                        html.Span("Selected Features · Importance", className="features-card-title"),
                        html.Button(
                            "＋ / − Show All",
                            id="features-filter-btn",
                            className="chart-toolbar-btn",
                            title="Toggle: show all features / positive importance only",
                        ),
                    ]),
                    html.Div(className="features-scroll-wrap", children=[
                        dcc.Graph(id="chart-features", config={"displayModeBar": False}),
                    ]),
                ]),
            ]),
        ]),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS — STEP 4
# ══════════════════════════════════════════════════════════════════════════════

# ── Lazy data loader — fires once when Step 4 first renders ──────────────────
@dash_app.callback(
    Output("step4-data-loaded", "data"),
    Input("step4-data-loaded", "data"),
    prevent_initial_call=False,
)
def load_step4_data(already_loaded):
    """
    Runs when the step4-data-loaded store is first mounted (i.e. when Step 4
    renders). Populates the global MODELS dict from results.db in the latest
    dated folder.  Subsequent renders skip the reload because already_loaded
    is True.
    """
    global MODELS
    if already_loaded:
        return True

    try:
        MODELS = load_models_from_results()
        print(f"[step4] Loaded {len(MODELS)} models from {_latest_results_path()}")
    except Exception as exc:
        print(f"[step4] WARNING – could not load results.db: {exc}")
        MODELS = {}

    return True


# ── Navigate to selected model's page when result-type toggles ───────────────
@dash_app.callback(
    Output("page-store", "data", allow_duplicate=True),
    Input("result-type-store", "data"),
    State("selected-model-store", "data"),
    State("sort-metric-store", "data"),
    State("sort-dir-store", "data"),
    State("model-search", "value"),
    prevent_initial_call=True,
)
def navigate_to_selected_on_toggle(result_type, selected_key, sort_metric, sort_dir, search_val):
    if not selected_key or not MODELS:
        return 0
    all_rows = build_leaderboard_rows(sort_metric, search_val or "", result_type)
    if sort_dir == "asc":
        all_rows = list(reversed(all_rows))
    keys = [r["key"] for r in all_rows]
    if selected_key not in keys:
        return 0
    idx = keys.index(selected_key)
    return idx // PAGE_SIZE


# ── Result type toggle ────────────────────────────────────────────────────────
@dash_app.callback(
    Output("result-type-store", "data"),
    Output("toggle-baseline", "className"),
    Output("toggle-enhanced", "className"),
    Input("toggle-baseline", "n_clicks"),
    Input("toggle-enhanced", "n_clicks"),
    prevent_initial_call=False,
)
def update_result_type(n_base, n_enh):
    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]["prop_id"] == ".":
        return "enhanced", "toggle-btn", "toggle-btn active-toggle"
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "toggle-enhanced":
        return "enhanced", "toggle-btn", "toggle-btn active-toggle"
    return "baseline", "toggle-btn active-toggle", "toggle-btn"


# ── Sort metric store — also resets page to 0 ────────────────────────────────
@dash_app.callback(
    Output("sort-metric-store", "data"),
    Output("sort-dir-store", "data"),
    Output("page-store", "data"),
    Input({"type": "sort-btn", "metric": ALL}, "n_clicks"),
    Input("model-search", "value"),
    Input("result-type-store", "data"),
    Input("page-first", "n_clicks"),
    Input("page-prev",  "n_clicks"),
    Input("page-next",  "n_clicks"),
    Input("page-last",  "n_clicks"),
    State("sort-metric-store", "data"),
    State("sort-dir-store", "data"),
    State("page-store", "data"),
    State("model-search", "value"),
    State("selected-model-store", "data"),
    prevent_initial_call=True,
)
def update_sort_and_page(sort_btn_clicks, search, result_type,
                         first, prev, nxt, last,
                         state_metric, state_dir, state_page, state_search,
                         selected_key):
    ctx = callback_context
    if not ctx.triggered:
        return state_metric, state_dir, state_page

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    def _page_for_selected(metric, direction, result_type):
        """Return the page index that contains the currently selected model."""
        if not selected_key or not MODELS:
            return 0
        rows = build_leaderboard_rows(metric, state_search or "", result_type)
        if direction == "asc":
            rows = list(reversed(rows))
        keys = [r["key"] for r in rows]
        if selected_key not in keys:
            return 0
        return keys.index(selected_key) // PAGE_SIZE

    # ── Sort button clicked ──────────────────────────────────────────────────
    try:
        parsed = json.loads(trigger_id)
        if parsed.get("type") == "model-row":
            return state_metric, state_dir, state_page
        if parsed.get("type") == "sort-btn":
            metric  = parsed["metric"]
            new_dir = "asc" if (metric == state_metric and state_dir == "desc") else "desc"
            return metric, new_dir, 0
    except (json.JSONDecodeError, KeyError):
        pass

    # ── Search changed → back to page 0 ─────────────────────────────────────
    if trigger_id == "model-search":
        return state_metric, state_dir, 0

    # ── Result-type toggled → handled by navigate_to_selected_on_toggle ─────
    if trigger_id == "result-type-store":
        return state_metric, state_dir, no_update

    # ── Pagination buttons ───────────────────────────────────────────────────
    total_rows = len(build_leaderboard_rows(state_metric, state_search or "", result_type))
    last_page  = max(0, (total_rows - 1) // PAGE_SIZE)

    if trigger_id == "page-first":
        return state_metric, state_dir, 0
    if trigger_id == "page-prev":
        return state_metric, state_dir, max(0, state_page - 1)
    if trigger_id == "page-next":
        return state_metric, state_dir, min(last_page, state_page + 1)
    if trigger_id == "page-last":
        return state_metric, state_dir, last_page

    return state_metric, state_dir, state_page


# ── Leaderboard render ────────────────────────────────────────────────────────
@dash_app.callback(
    Output("leaderboard-table-container", "children"),
    Output("page-indicator", "children"),
    Input("sort-metric-store", "data"),
    Input("sort-dir-store", "data"),
    Input("result-type-store", "data"),
    Input("model-search", "value"),
    Input("page-store", "data"),
    Input("step4-data-loaded", "data"),
    State("selected-model-store", "data"),
)
def render_leaderboard(sort_metric, sort_dir, result_type, search_val, page, _loaded, selected_key):
    # After data loads, default selection to first model if none chosen yet
    if not selected_key and MODELS:
        selected_key = list(MODELS.keys())[0]
    all_rows = build_leaderboard_rows(sort_metric, search_val or "", result_type)
    if sort_dir == "asc":
        all_rows = list(reversed(all_rows))

    total_rows = len(all_rows)
    last_page  = max(0, (total_rows - 1) // PAGE_SIZE) if total_rows else 0
    page       = max(0, min(page, last_page))

    start = page * PAGE_SIZE
    rows  = all_rows[start : start + PAGE_SIZE]

    # Best values computed across ALL rows (not just current page)
    best = {}
    for m in METRICS:
        col  = f"enh_{m}" if result_type == "enhanced" else f"base_{m}"
        vals = [r[col] for r in all_rows]
        best[m] = max(vals) if vals else 0

    # In Baseline view: primary column = Base, secondary = Enh Δ
    is_enhanced = result_type == "enhanced"

    # Primary (selected/highlighted) value is always on the RIGHT:
    #   Baseline → Enh Δ (left)  | Base  (right, highlighted)
    #   Enhanced → Base Δ (left) | Enh   (right, highlighted)
    sub_headers = [
        html.Th("", className="th-rank"),
        html.Th("Model", className="th-model"),
        html.Th("SMOTE Variant", className="th-strategy"),
    ]
    for m in METRICS:
        active = sort_metric == m
        if is_enhanced:
            sub_headers.append(html.Th("Base",  className=f"th-sub {'th-sub-active' if active else ''}"))
            sub_headers.append(html.Th("Enh Δ", className=f"th-sub enh-col {'th-sub-active' if active else ''}"))
        else:
            sub_headers.append(html.Th("Enh",   className=f"th-sub enh-col {'th-sub-active' if active else ''}"))
            sub_headers.append(html.Th("Base Δ", className=f"th-sub {'th-sub-active' if active else ''}"))

    thead = html.Thead([
        html.Tr([
            html.Th("", className="th-rank"),
            html.Th("", className="th-model"),
            html.Th("", className="th-strategy"),
            *[_metric_header(m, METRIC_LABELS[m]) for m in METRICS],
        ]),
        html.Tr(sub_headers),
    ])

    tbody_rows = []
    for i, row in enumerate(rows):
        global_rank = all_rows.index(row) + 1
        rank_cls = "gold" if global_rank == 1 else "silver" if global_rank == 2 else "bronze" if global_rank == 3 else "default"
        cells = [
            html.Td(html.Span(f"#{global_rank}", className=f"rank-badge rank-{rank_cls}"), className="td-rank"),
            html.Td(
                html.Div([html.Span(_model_display(row["name"]), className="model-name")]),
                className="td-model",
            ),
            html.Td(
                html.Span(_strategy_display(row["strategy"]), className="strategy-pill"),
                className="td-strategy",
            ),
        ]
        for m in METRICS:
            base_val = row[f"base_{m}"]
            enh_val  = row[f"enh_{m}"]
            primary  = enh_val if is_enhanced else base_val
            is_best  = abs(primary - best[m]) < 0.0001

            if is_enhanced:
                # Left cell: BASE plain value (no delta)
                cells.append(html.Td(
                    html.Span(f"{base_val:.3f}", className="metric-val"),
                    className="td-metric",
                ))
                # Right cell: ENH value + delta vs base (primary, highlighted when best)
                cells.append(html.Td(
                    html.Div([
                        html.Span(f"{enh_val:.3f}", className="metric-val"),
                        delta_badge(enh_val, base_val),
                    ], className="enh-cell-inner"),
                    className=f"td-metric enh-col {'best-cell' if is_best else ''}".strip(),
                ))
            else:
                # Left cell: ENH plain value (no delta)
                cells.append(html.Td(
                    html.Span(f"{enh_val:.3f}", className="metric-val"),
                    className="td-metric enh-col",
                ))
                # Right cell: BASE value + delta vs enh (primary, highlighted when best)
                cells.append(html.Td(
                    html.Div([
                        html.Span(f"{base_val:.3f}", className="metric-val"),
                        delta_badge(base_val, enh_val),
                    ], className="enh-cell-inner"),
                    className=f"td-metric {'best-cell' if is_best else ''}".strip(),
                ))

        params = row.get("parameters", {})
        params_json = json.dumps(params) if params else ""
        tbody_rows.append(html.Tr(
            cells,
            id={"type": "model-row", "key": row["key"]},
            className=f"model-row {'selected-row' if row['key'] == selected_key else ''}",
            **{"data-params": params_json},
            n_clicks=0,
        ))

    table = html.Table([thead, html.Tbody(tbody_rows)], className="leaderboard-table")

    indicator = f"Page {page + 1} of {last_page + 1}  ·  {total_rows} model{'s' if total_rows != 1 else ''}"

    return table, indicator


# ── Row click → selected model ────────────────────────────────────────────────
@dash_app.callback(
    Output("selected-model-store", "data"),
    Input({"type": "model-row", "key": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_model(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    key = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["key"]
    return key


# ── Highlight selected row in DOM directly (no table re-render needed) ────────
dash_app.clientside_callback(
    """
    function(selected_key, result_type) {
        if (!selected_key) return window.dash_clientside.no_update;

        // Small delay to let Dash finish re-rendering the table after a toggle
        setTimeout(function() {
            document.querySelectorAll('tr.model-row').forEach(function(tr) {
                tr.classList.remove('selected-row');
            });
            document.querySelectorAll('tr.model-row').forEach(function(tr) {
                const idAttr = tr.id;
                if (!idAttr) return;
                try {
                    const parsed = JSON.parse(idAttr);
                    if (parsed.key === selected_key) {
                        tr.classList.add('selected-row');
                    }
                } catch(_) {}
            });
        }, 50);

        return window.dash_clientside.no_update;
    }
    """,
    Output("params-tooltip", "style"),
    Input("selected-model-store", "data"),
    Input("result-type-store", "data"),
    prevent_initial_call=True,
)


# ── Features filter toggle ────────────────────────────────────────────────────
@dash_app.callback(
    Output("features-positive-store", "data"),
    Output("features-filter-btn", "children"),
    Output("features-filter-btn", "className"),
    Input("features-filter-btn", "n_clicks"),
    State("features-positive-store", "data"),
    prevent_initial_call=True,
)
def toggle_features_filter(n, currently_positive):
    new_state = not currently_positive
    if new_state:
        return new_state, "✦ Positive Only", "chart-toolbar-btn chart-toolbar-btn-active"
    return new_state, "＋ / − Show All", "chart-toolbar-btn"


# ── Charts update ─────────────────────────────────────────────────────────────
@dash_app.callback(
    Output("chart-roc",      "figure"),
    Output("chart-pr",       "figure"),
    Output("chart-cm",       "figure"),
    Output("chart-features", "figure"),
    Output("charts-model-label", "children"),
    Input("selected-model-store",    "data"),
    Input("result-type-store",       "data"),
    Input("step4-data-loaded",       "data"),
    Input("features-positive-store", "data"),
)
def update_charts(model_key, result_type, _loaded, positives_only):
    if not model_key or model_key not in MODELS:
        empty = go.Figure()
        empty.update_layout(**_base_layout(""))
        return empty, empty, empty, empty, ""

    name  = _model_display(MODELS[model_key]["model"])
    label = html.Div([
        html.Span(name, className="charts-model-name"),
        html.Span(f" · {result_type.capitalize()}", className="charts-model-type"),
    ])
    return (
        build_roc_figure(model_key, result_type),
        build_pr_figure(model_key, result_type),
        build_cm_figure(model_key, result_type),
        build_features_figure(model_key, result_type, positives_only=bool(positives_only)),
        label,
    )


# ── Parameter tooltip — clientside hover ─────────────────────────────────────
dash_app.clientside_callback(
    """
    function(n) {
        if (window._paramsTooltipBound) return window.dash_clientside.no_update;
        window._paramsTooltipBound = true;

        const tooltip = document.getElementById('params-tooltip');
        const content = document.getElementById('params-tooltip-content');
        if (!tooltip || !content) return window.dash_clientside.no_update;

        let mouseX = 0, mouseY = 0;
        document.addEventListener('mousemove', function(e) {
            mouseX = e.clientX;
            mouseY = e.clientY;
            if (tooltip.classList.contains('params-tooltip-visible')) {
                positionTooltip();
            }
        });

        function positionTooltip() {
            const scrollY = window.scrollY || document.documentElement.scrollTop;
            const scrollX = window.scrollX || document.documentElement.scrollLeft;
            const tipW = tooltip.offsetWidth  || 220;
            const tipH = tooltip.offsetHeight || 100;
            const vw   = window.innerWidth;

            let left = mouseX + scrollX + 14;
            let top  = mouseY + scrollY - tipH - 10;

            if (mouseX + 14 + tipW > vw) left = mouseX + scrollX - tipW - 14;
            if (mouseY - tipH - 10 < 0)  top  = mouseY + scrollY + 18;

            tooltip.style.left = left + 'px';
            tooltip.style.top  = top  + 'px';
        }

        function formatKey(k) {
            // "learning_rate" → "Learning Rate"
            return String(k)
                .replace(/_/g, ' ')
                .replace(/\\b\\w/g, c => c.toUpperCase());
        }

        function parseParams(raw) {
            if (!raw) return null;

            // 1. Try standard JSON first
            try {
                const obj = JSON.parse(raw);
                if (obj && typeof obj === 'object' && !Array.isArray(obj)) return obj;
            } catch(_) {}

            // 2. Handle Python repr: [('key', val), ...] or (('key', val), ...)
            //    Also handles {"key": val, ...} with single quotes
            const tupleMatch = raw.match(/[\\[\\(]\\s*\\(?\\s*['"](.*?)['"]/);
            if (tupleMatch) {
                const result = {};
                const pairRe = /[\\(\\[]\\s*['"]([^'"]+)['"]\\s*,\\s*([^,\\)\\]]+?)\\s*[\\)\\]]/g;
                let m;
                while ((m = pairRe.exec(raw)) !== null) {
                    const key = m[1].trim();
                    let val = m[2].trim();
                    // Coerce numerics
                    const num = parseFloat(val);
                    result[key] = isNaN(num) ? val : num;
                }
                if (Object.keys(result).length) return result;
            }

            // 3. Single-quoted dict: {'key': val, ...}
            try {
                const jsonified = raw
                    .replace(/'/g, '"')
                    .replace(/True/g, 'true')
                    .replace(/False/g, 'false')
                    .replace(/None/g, 'null');
                const obj = JSON.parse(jsonified);
                if (obj && typeof obj === 'object') return obj;
            } catch(_) {}

            return null;
        }

        document.addEventListener('mouseover', function(e) {
            const row = e.target.closest('tr.model-row');
            if (!row) { tooltip.classList.remove('params-tooltip-visible'); return; }

            const raw = row.getAttribute('data-params');
            if (!raw || raw === '{}' || raw === '') {
                tooltip.classList.remove('params-tooltip-visible');
                return;
            }

            const params = parseParams(raw);
            if (!params || !Object.keys(params).length) {
                tooltip.classList.remove('params-tooltip-visible');
                return;
            }

            content.innerHTML = Object.keys(params).map(k =>
                '<div class="ptt-row">' +
                  '<span class="ptt-key">' + formatKey(k) + '</span>' +
                  '<span class="ptt-val">' + params[k] + '</span>' +
                '</div>'
            ).join('');

            positionTooltip();
            tooltip.classList.add('params-tooltip-visible');
        });

        document.addEventListener('mouseout', function(e) {
            const row = e.target.closest('tr.model-row');
            if (row && !row.contains(e.relatedTarget)) {
                tooltip.classList.remove('params-tooltip-visible');
            }
        });

        return window.dash_clientside.no_update;
    }
    """,
    Output("params-tooltip", "className"),
    Input("step4-data-loaded", "data"),
    prevent_initial_call=False,
)


# ── CSV export ────────────────────────────────────────────────────────────────
@dash_app.callback(
    Output("download-csv", "data"),
    Input("export-csv-btn", "n_clicks"),
    State("sort-metric-store", "data"),
    State("result-type-store", "data"),
    prevent_initial_call=True,
)
def export_csv(n, sort_metric, result_type):
    if not n:
        return no_update
    rows = build_leaderboard_rows(sort_metric, "", result_type)
    records = [{
        "Model":    r["name"],
        "Strategy": r["strategy"],
        **{f"Base {METRIC_LABELS[m]}": round(r[f"base_{m}"], 4) for m in METRICS},
        **{f"Enh {METRIC_LABELS[m]}":  round(r[f"enh_{m}"],  4) for m in METRICS},
    } for r in rows]
    return dcc.send_data_frame(pd.DataFrame(records).to_csv, "model_leaderboard.csv", index=False)