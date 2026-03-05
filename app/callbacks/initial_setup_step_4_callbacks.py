import json
import pandas as pd
import plotly.graph_objects as go
import numpy as np

import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, dcc, no_update, callback_context

from app import dash_app

# ══════════════════════════════════════════════════════════════════════════════
#  DUMMY DATA
# ══════════════════════════════════════════════════════════════════════════════

DUMMY_MODELS = {
    "random_forest__none__none": {
        "model": "Random Forest",
        "balance_strategy": "none",
        "baseline": {
            "evaluation": {
                "metrics": {
                    "accuracy": 0.891, "precision_macro": 0.874,
                    "recall_macro": 0.862, "f1_macro": 0.868, "roc_auc_macro": 0.943,
                },
                "charts": {
                    "confusion_matrix": [[210, 18, 7], [14, 198, 12], [9, 11, 221]],
                    "roc_curve": {
                        "0": {"fpr": [0, 0.02, 0.05, 0.1, 1], "tpr": [0, 0.72, 0.88, 0.94, 1]},
                        "1": {"fpr": [0, 0.03, 0.06, 0.12, 1], "tpr": [0, 0.68, 0.85, 0.92, 1]},
                        "2": {"fpr": [0, 0.01, 0.04, 0.08, 1], "tpr": [0, 0.75, 0.91, 0.96, 1]},
                    },
                    "pr_curve": {
                        "0": {"precision": [1, 0.91, 0.87, 0.82, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "1": {"precision": [1, 0.88, 0.83, 0.78, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "2": {"precision": [1, 0.93, 0.89, 0.84, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                    },
                },
            },
            "features": ["payment_history", "credit_score", "dtp_30", "dtp_60", "balance_ratio"],
        },
        "enhanced": {
            "evaluation": {
                "metrics": {
                    "accuracy": 0.912, "precision_macro": 0.901,
                    "recall_macro": 0.889, "f1_macro": 0.895, "roc_auc_macro": 0.961,
                },
                "charts": {
                    "confusion_matrix": [[218, 12, 5], [10, 207, 7], [6, 8, 227]],
                    "roc_curve": {
                        "0": {"fpr": [0, 0.01, 0.04, 0.08, 1], "tpr": [0, 0.78, 0.91, 0.96, 1]},
                        "1": {"fpr": [0, 0.02, 0.05, 0.1, 1], "tpr": [0, 0.74, 0.89, 0.94, 1]},
                        "2": {"fpr": [0, 0.01, 0.03, 0.06, 1], "tpr": [0, 0.81, 0.94, 0.98, 1]},
                    },
                    "pr_curve": {
                        "0": {"precision": [1, 0.94, 0.91, 0.87, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "1": {"precision": [1, 0.91, 0.87, 0.83, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "2": {"precision": [1, 0.96, 0.92, 0.88, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                    },
                },
            },
            "features": ["payment_history", "credit_score", "dtp_30", "dtp_60", "dtp_90", "balance_ratio", "survival_score"],
        },
    },
    "xgboost__none__none": {
        "model": "XGBoost",
        "balance_strategy": "none",
        "baseline": {
            "evaluation": {
                "metrics": {
                    "accuracy": 0.903, "precision_macro": 0.889,
                    "recall_macro": 0.877, "f1_macro": 0.883, "roc_auc_macro": 0.956,
                },
                "charts": {
                    "confusion_matrix": [[214, 15, 6], [12, 203, 9], [7, 10, 224]],
                    "roc_curve": {
                        "0": {"fpr": [0, 0.02, 0.04, 0.09, 1], "tpr": [0, 0.74, 0.90, 0.95, 1]},
                        "1": {"fpr": [0, 0.02, 0.05, 0.11, 1], "tpr": [0, 0.71, 0.87, 0.93, 1]},
                        "2": {"fpr": [0, 0.01, 0.03, 0.07, 1], "tpr": [0, 0.77, 0.93, 0.97, 1]},
                    },
                    "pr_curve": {
                        "0": {"precision": [1, 0.92, 0.89, 0.84, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "1": {"precision": [1, 0.89, 0.85, 0.80, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "2": {"precision": [1, 0.94, 0.91, 0.86, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                    },
                },
            },
            "features": ["credit_score", "dtp_30", "dtp_60", "payment_history", "loan_amount"],
        },
        "enhanced": {
            "evaluation": {
                "metrics": {
                    "accuracy": 0.928, "precision_macro": 0.917,
                    "recall_macro": 0.908, "f1_macro": 0.912, "roc_auc_macro": 0.971,
                },
                "charts": {
                    "confusion_matrix": [[222, 10, 3], [8, 211, 5], [4, 6, 231]],
                    "roc_curve": {
                        "0": {"fpr": [0, 0.01, 0.03, 0.07, 1], "tpr": [0, 0.81, 0.93, 0.97, 1]},
                        "1": {"fpr": [0, 0.01, 0.04, 0.09, 1], "tpr": [0, 0.77, 0.91, 0.95, 1]},
                        "2": {"fpr": [0, 0.01, 0.02, 0.05, 1], "tpr": [0, 0.84, 0.96, 0.99, 1]},
                    },
                    "pr_curve": {
                        "0": {"precision": [1, 0.95, 0.92, 0.89, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "1": {"precision": [1, 0.92, 0.89, 0.85, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "2": {"precision": [1, 0.97, 0.94, 0.90, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                    },
                },
            },
            "features": ["credit_score", "dtp_30", "dtp_60", "dtp_90", "payment_history", "loan_amount", "survival_score", "hazard_ratio"],
        },
    },
    "decision_tree__none__none": {
        "model": "Decision Tree",
        "balance_strategy": "none",
        "baseline": {
            "evaluation": {
                "metrics": {
                    "accuracy": 0.821, "precision_macro": 0.809,
                    "recall_macro": 0.795, "f1_macro": 0.802, "roc_auc_macro": 0.871,
                },
                "charts": {
                    "confusion_matrix": [[192, 28, 15], [22, 181, 21], [18, 19, 204]],
                    "roc_curve": {
                        "0": {"fpr": [0, 0.05, 0.10, 0.18, 1], "tpr": [0, 0.61, 0.78, 0.87, 1]},
                        "1": {"fpr": [0, 0.06, 0.12, 0.20, 1], "tpr": [0, 0.58, 0.74, 0.84, 1]},
                        "2": {"fpr": [0, 0.04, 0.09, 0.15, 1], "tpr": [0, 0.64, 0.81, 0.90, 1]},
                    },
                    "pr_curve": {
                        "0": {"precision": [1, 0.84, 0.79, 0.73, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "1": {"precision": [1, 0.81, 0.76, 0.70, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "2": {"precision": [1, 0.87, 0.82, 0.76, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                    },
                },
            },
            "features": ["credit_score", "payment_history", "loan_amount"],
        },
        "enhanced": {
            "evaluation": {
                "metrics": {
                    "accuracy": 0.847, "precision_macro": 0.836,
                    "recall_macro": 0.821, "f1_macro": 0.828, "roc_auc_macro": 0.899,
                },
                "charts": {
                    "confusion_matrix": [[199, 22, 14], [18, 189, 17], [14, 16, 211]],
                    "roc_curve": {
                        "0": {"fpr": [0, 0.04, 0.08, 0.15, 1], "tpr": [0, 0.65, 0.82, 0.90, 1]},
                        "1": {"fpr": [0, 0.05, 0.10, 0.17, 1], "tpr": [0, 0.62, 0.78, 0.87, 1]},
                        "2": {"fpr": [0, 0.03, 0.07, 0.12, 1], "tpr": [0, 0.68, 0.85, 0.93, 1]},
                    },
                    "pr_curve": {
                        "0": {"precision": [1, 0.87, 0.83, 0.77, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "1": {"precision": [1, 0.84, 0.79, 0.73, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "2": {"precision": [1, 0.90, 0.86, 0.80, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                    },
                },
            },
            "features": ["credit_score", "payment_history", "loan_amount", "dtp_30", "survival_score"],
        },
    },
    "ada_boost__none__none": {
        "model": "AdaBoost",
        "balance_strategy": "none",
        "baseline": {
            "evaluation": {
                "metrics": {
                    "accuracy": 0.856, "precision_macro": 0.841,
                    "recall_macro": 0.829, "f1_macro": 0.835, "roc_auc_macro": 0.912,
                },
                "charts": {
                    "confusion_matrix": [[201, 24, 10], [18, 193, 13], [11, 15, 215]],
                    "roc_curve": {
                        "0": {"fpr": [0, 0.03, 0.07, 0.13, 1], "tpr": [0, 0.67, 0.83, 0.91, 1]},
                        "1": {"fpr": [0, 0.04, 0.08, 0.15, 1], "tpr": [0, 0.63, 0.80, 0.88, 1]},
                        "2": {"fpr": [0, 0.02, 0.06, 0.11, 1], "tpr": [0, 0.70, 0.86, 0.93, 1]},
                    },
                    "pr_curve": {
                        "0": {"precision": [1, 0.87, 0.83, 0.78, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "1": {"precision": [1, 0.84, 0.80, 0.75, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "2": {"precision": [1, 0.90, 0.86, 0.81, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                    },
                },
            },
            "features": ["credit_score", "dtp_30", "payment_history"],
        },
        "enhanced": {
            "evaluation": {
                "metrics": {
                    "accuracy": 0.878, "precision_macro": 0.865,
                    "recall_macro": 0.854, "f1_macro": 0.859, "roc_auc_macro": 0.931,
                },
                "charts": {
                    "confusion_matrix": [[208, 19, 8], [14, 201, 9], [8, 12, 221]],
                    "roc_curve": {
                        "0": {"fpr": [0, 0.02, 0.05, 0.11, 1], "tpr": [0, 0.71, 0.87, 0.93, 1]},
                        "1": {"fpr": [0, 0.03, 0.07, 0.13, 1], "tpr": [0, 0.67, 0.84, 0.91, 1]},
                        "2": {"fpr": [0, 0.02, 0.04, 0.09, 1], "tpr": [0, 0.74, 0.90, 0.96, 1]},
                    },
                    "pr_curve": {
                        "0": {"precision": [1, 0.90, 0.86, 0.82, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "1": {"precision": [1, 0.87, 0.83, 0.79, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "2": {"precision": [1, 0.93, 0.89, 0.85, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                    },
                },
            },
            "features": ["credit_score", "dtp_30", "dtp_60", "payment_history", "survival_score"],
        },
    },
    "nn_mlp__none__none": {
        "model": "MLP Neural Net",
        "balance_strategy": "none",
        "baseline": {
            "evaluation": {
                "metrics": {
                    "accuracy": 0.879, "precision_macro": 0.863,
                    "recall_macro": 0.851, "f1_macro": 0.857, "roc_auc_macro": 0.934,
                },
                "charts": {
                    "confusion_matrix": [[207, 20, 8], [15, 199, 10], [9, 12, 220]],
                    "roc_curve": {
                        "0": {"fpr": [0, 0.02, 0.05, 0.11, 1], "tpr": [0, 0.71, 0.87, 0.93, 1]},
                        "1": {"fpr": [0, 0.03, 0.07, 0.13, 1], "tpr": [0, 0.67, 0.83, 0.91, 1]},
                        "2": {"fpr": [0, 0.02, 0.04, 0.09, 1], "tpr": [0, 0.74, 0.90, 0.96, 1]},
                    },
                    "pr_curve": {
                        "0": {"precision": [1, 0.89, 0.85, 0.80, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "1": {"precision": [1, 0.86, 0.82, 0.77, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "2": {"precision": [1, 0.91, 0.87, 0.82, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                    },
                },
            },
            "features": ["credit_score", "dtp_30", "dtp_60", "payment_history", "balance_ratio"],
        },
        "enhanced": {
            "evaluation": {
                "metrics": {
                    "accuracy": 0.919, "precision_macro": 0.908,
                    "recall_macro": 0.897, "f1_macro": 0.902, "roc_auc_macro": 0.966,
                },
                "charts": {
                    "confusion_matrix": [[220, 11, 4], [9, 209, 6], [5, 7, 229]],
                    "roc_curve": {
                        "0": {"fpr": [0, 0.01, 0.03, 0.07, 1], "tpr": [0, 0.79, 0.92, 0.97, 1]},
                        "1": {"fpr": [0, 0.02, 0.04, 0.09, 1], "tpr": [0, 0.75, 0.90, 0.95, 1]},
                        "2": {"fpr": [0, 0.01, 0.02, 0.05, 1], "tpr": [0, 0.82, 0.95, 0.98, 1]},
                    },
                    "pr_curve": {
                        "0": {"precision": [1, 0.94, 0.91, 0.87, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "1": {"precision": [1, 0.91, 0.88, 0.84, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                        "2": {"precision": [1, 0.96, 0.93, 0.89, 0], "recall": [0, 0.3, 0.6, 0.9, 1]},
                    },
                },
            },
            "features": ["credit_score", "dtp_30", "dtp_60", "dtp_90", "payment_history", "balance_ratio", "survival_score", "hazard_ratio"],
        },
    },
}

METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision_macro": "Precision",
    "recall_macro": "Recall",
    "f1_macro": "F1",
    "roc_auc_macro": "ROC-AUC",
}
CHART_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY THEME  —  light mode, no duplicate axis keys
# ══════════════════════════════════════════════════════════════════════════════

def _base_layout(title_text):
    """Return a clean light-mode layout dict. Axis overrides applied per-figure."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        font=dict(family="'IBM Plex Mono', monospace", color="#374151", size=10),
        margin=dict(l=44, r=16, t=36, b=36),
        title=dict(
            text=title_text,
            font=dict(family="'DM Serif Display', serif", size=13, color="#111827"),
            x=0,
            xanchor="left",
            pad=dict(l=4),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e5e7eb",
            borderwidth=1,
            font=dict(size=9),
        ),
        xaxis=dict(
            gridcolor="#e5e7eb",
            linecolor="#d1d5db",
            tickfont=dict(size=9),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#e5e7eb",
            linecolor="#d1d5db",
            tickfont=dict(size=9),
            zeroline=False,
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_metrics(model_key, result_type):
    return DUMMY_MODELS[model_key][result_type]["evaluation"]["metrics"]


def delta_badge(val, ref):
    diff = val - ref
    if abs(diff) < 0.0005:
        return html.Span("—", className="delta-neutral")
    sign = "▲" if diff > 0 else "▼"
    cls = "delta-up" if diff > 0 else "delta-down"
    return html.Span(f"{sign}{abs(diff):.3f}", className=cls)


def build_leaderboard_rows(sort_metric, search_val, result_type):
    rows = []
    for key, data in DUMMY_MODELS.items():
        base_m = get_metrics(key, "baseline")
        enh_m  = get_metrics(key, "enhanced")
        rows.append({
            "key": key,
            "name": data["model"],
            "strategy": data["balance_strategy"],
            **{f"base_{m}": base_m.get(m, 0) for m in METRICS},
            **{f"enh_{m}":  enh_m.get(m, 0)  for m in METRICS},
            "sort_val": (enh_m if result_type == "enhanced" else base_m).get(sort_metric, 0),
        })
    if search_val:
        rows = [r for r in rows if search_val.lower() in r["name"].lower()]
    rows.sort(key=lambda x: x["sort_val"], reverse=True)
    return rows


def build_roc_figure(model_key, result_type):
    roc = DUMMY_MODELS[model_key][result_type]["evaluation"]["charts"].get("roc_curve", {})
    auc = get_metrics(model_key, result_type).get("roc_auc_macro", 0)
    layout = _base_layout(f"ROC Curve · AUC {auc:.3f}")
    layout["xaxis"]["title"] = "False Positive Rate"
    layout["xaxis"]["range"] = [0, 1]
    layout["yaxis"]["title"] = "True Positive Rate"
    layout["yaxis"]["range"] = [0, 1]

    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="#9ca3af", dash="dash", width=1))
    for i, cls in enumerate(sorted(roc.keys())):
        color = CHART_COLORS[i % len(CHART_COLORS)]
        fig.add_trace(go.Scatter(
            x=roc[cls]["fpr"], y=roc[cls]["tpr"],
            mode="lines", name=f"Class {cls}",
            line=dict(color=color, width=2),
        ))
    fig.update_layout(**layout)
    return fig


def build_pr_figure(model_key, result_type):
    pr = DUMMY_MODELS[model_key][result_type]["evaluation"]["charts"].get("pr_curve", {})
    layout = _base_layout("Precision–Recall Curve")
    layout["xaxis"]["title"] = "Recall"
    layout["xaxis"]["range"] = [0, 1]
    layout["yaxis"]["title"] = "Precision"
    layout["yaxis"]["range"] = [0, 1.05]

    fig = go.Figure()
    for i, cls in enumerate(sorted(pr.keys())):
        color = CHART_COLORS[i % len(CHART_COLORS)]
        fig.add_trace(go.Scatter(
            x=pr[cls]["recall"], y=pr[cls]["precision"],
            mode="lines", name=f"Class {cls}",
            line=dict(color=color, width=2),
        ))
    fig.update_layout(**layout)
    return fig


def build_cm_figure(model_key, result_type):
    cm = DUMMY_MODELS[model_key][result_type]["evaluation"]["charts"]["confusion_matrix"]
    cm_arr = np.array(cm)
    total = cm_arr.sum(axis=1, keepdims=True)
    pct = cm_arr / total * 100
    labels = [[f"{cm_arr[i][j]}<br>{pct[i][j]:.1f}%"
               for j in range(len(cm[0]))] for i in range(len(cm))]

    layout = _base_layout("Confusion Matrix")
    layout["xaxis"]["title"] = "Predicted"
    layout["yaxis"]["title"] = "Actual"

    fig = go.Figure(go.Heatmap(
        z=cm_arr,
        text=labels,
        texttemplate="%{text}",
        textfont=dict(size=10, family="'IBM Plex Mono', monospace", color="#111827"),
        colorscale=[[0, "#eff6ff"], [0.5, "#93c5fd"], [1, "#1d4ed8"]],
        showscale=False,
    ))
    fig.update_layout(**layout)
    return fig


def build_features_figure(model_key, result_type):
    features = DUMMY_MODELS[model_key][result_type].get("features", [])
    importance = sorted(
        [round(1 - i * 0.1 + np.random.uniform(-0.03, 0.03), 3) for i in range(len(features))],
        reverse=True,
    )
    colors = [CHART_COLORS[0] if i == 0 else "#93c5fd" for i in range(len(features))]

    layout = _base_layout("Selected Features · Importance")
    layout["xaxis"]["title"] = "Importance Score"
    layout["yaxis"]["autorange"] = "reversed"
    layout["margin"]["l"] = 120

    fig = go.Figure(go.Bar(
        x=importance, y=features,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in importance],
        textposition="outside",
        textfont=dict(size=9, color="#6b7280"),
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
                html.P("Comparative performance across baseline and survival-enhanced pipelines.",
                       className="dash-subtitle"),
            ]),
            html.Div(className="dash-header-right", children=[
                html.Div(className="result-toggle-wrap", children=[
                    html.Span("View:", className="toggle-label"),
                    html.Div(className="result-toggle", children=[
                        html.Button("Baseline", id="toggle-baseline", className="toggle-btn active-toggle"),
                        html.Button("Enhanced", id="toggle-enhanced", className="toggle-btn"),
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
        dcc.Store(id="sort-metric-store", data="f1_macro"),
        dcc.Store(id="sort-dir-store", data="desc"),
        dcc.Store(id="result-type-store", data="baseline"),
        dcc.Store(id="selected-model-store", data=list(DUMMY_MODELS.keys())[0]),

        # ── Leaderboard table ────────────────────────────────────────────────
        html.Div(className="leaderboard-wrap", children=[
            html.Div(id="leaderboard-table-container"),
        ]),

        # ── Chart section ────────────────────────────────────────────────────
        html.Div(className="charts-section", children=[
            html.Div(id="charts-model-label", className="charts-model-label"),
            html.Div(className="charts-grid", children=[
                html.Div(className="chart-card", children=[
                    dcc.Graph(id="chart-roc", config={"displayModeBar": False}),
                ]),
                html.Div(className="chart-card", children=[
                    dcc.Graph(id="chart-pr", config={"displayModeBar": False}),
                ]),
                html.Div(className="chart-card", children=[
                    dcc.Graph(id="chart-cm", config={"displayModeBar": False}),
                ]),
                html.Div(className="chart-card", children=[
                    dcc.Graph(id="chart-features", config={"displayModeBar": False}),
                ]),
            ]),
        ]),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS — STEP 4
# ══════════════════════════════════════════════════════════════════════════════

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
        return "baseline", "toggle-btn active-toggle", "toggle-btn"
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "toggle-enhanced":
        return "enhanced", "toggle-btn", "toggle-btn active-toggle"
    return "baseline", "toggle-btn active-toggle", "toggle-btn"


# ── Sort metric store ─────────────────────────────────────────────────────────
@dash_app.callback(
    Output("sort-metric-store", "data"),
    Output("sort-dir-store", "data"),
    [Input({"type": "sort-btn", "metric": m}, "n_clicks") for m in METRICS],
    State("sort-metric-store", "data"),
    State("sort-dir-store", "data"),
    prevent_initial_call=True,
)
def update_sort(*args):
    current_metric, current_dir = args[-2], args[-1]
    ctx = callback_context
    if not ctx.triggered:
        return current_metric, current_dir
    metric = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["metric"]
    new_dir = "asc" if (metric == current_metric and current_dir == "desc") else "desc"
    return metric, new_dir


# ── Leaderboard render ────────────────────────────────────────────────────────
@dash_app.callback(
    Output("leaderboard-table-container", "children"),
    Input("sort-metric-store", "data"),
    Input("sort-dir-store", "data"),
    Input("result-type-store", "data"),
    Input("model-search", "value"),
    Input("selected-model-store", "data"),
)
def render_leaderboard(sort_metric, sort_dir, result_type, search_val, selected_key):
    rows = build_leaderboard_rows(sort_metric, search_val or "", result_type)
    if sort_dir == "asc":
        rows = list(reversed(rows))

    best = {}
    for m in METRICS:
        col = f"enh_{m}" if result_type == "enhanced" else f"base_{m}"
        vals = [r[col] for r in rows]
        best[m] = max(vals) if vals else 0

    sub_headers = [
        html.Th("", className="th-rank"),
        html.Th("Model", className="th-model"),
        html.Th("Strategy", className="th-strategy"),
    ]
    for m in METRICS:
        active = sort_metric == m
        sub_headers.append(html.Th("Base", className=f"th-sub {'th-sub-active' if active else ''}"))
        sub_headers.append(html.Th("Enh ∆", className=f"th-sub enh-col {'th-sub-active' if active else ''}"))

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
    for rank, row in enumerate(rows, 1):
        rank_cls = "gold" if rank == 1 else "silver" if rank == 2 else "bronze" if rank == 3 else "default"
        cells = [
            html.Td(html.Span(f"#{rank}", className=f"rank-badge rank-{rank_cls}"), className="td-rank"),
            html.Td(row["name"], className="td-model"),
            html.Td(html.Span(row["strategy"], className="strategy-pill"), className="td-strategy"),
        ]
        for m in METRICS:
            base_val = row[f"base_{m}"]
            enh_val  = row[f"enh_{m}"]
            primary  = enh_val if result_type == "enhanced" else base_val
            is_best  = abs(primary - best[m]) < 0.0001
            cells.append(html.Td(
                html.Span(f"{base_val:.3f}", className="metric-val"),
                className=f"td-metric {'best-cell' if is_best and result_type == 'baseline' else ''}",
            ))
            cells.append(html.Td(
                html.Div([
                    html.Span(f"{enh_val:.3f}", className="metric-val"),
                    delta_badge(enh_val, base_val),
                ], className="enh-cell-inner"),
                className=f"td-metric enh-col {'best-cell' if is_best and result_type == 'enhanced' else ''}",
            ))

        tbody_rows.append(html.Tr(
            cells,
            id={"type": "model-row", "key": row["key"]},
            className=f"model-row {'selected-row' if row['key'] == selected_key else ''}",
            n_clicks=0,
        ))

    return html.Table([thead, html.Tbody(tbody_rows)], className="leaderboard-table")


# ── Row click → selected model ────────────────────────────────────────────────
@dash_app.callback(
    Output("selected-model-store", "data"),
    [Input({"type": "model-row", "key": k}, "n_clicks") for k in DUMMY_MODELS.keys()],
    prevent_initial_call=True,
)
def select_model(*args):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    key = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["key"]
    return key


# ── Charts update ─────────────────────────────────────────────────────────────
@dash_app.callback(
    Output("chart-roc", "figure"),
    Output("chart-pr", "figure"),
    Output("chart-cm", "figure"),
    Output("chart-features", "figure"),
    Output("charts-model-label", "children"),
    Input("selected-model-store", "data"),
    Input("result-type-store", "data"),
)
def update_charts(model_key, result_type):
    if not model_key or model_key not in DUMMY_MODELS:
        empty = go.Figure()
        empty.update_layout(**_base_layout(""))
        return empty, empty, empty, empty, ""

    name = DUMMY_MODELS[model_key]["model"]
    label = html.Div([
        html.Span(name, className="charts-model-name"),
        html.Span(f" · {result_type.capitalize()}", className="charts-model-type"),
    ])
    return (
        build_roc_figure(model_key, result_type),
        build_pr_figure(model_key, result_type),
        build_cm_figure(model_key, result_type),
        build_features_figure(model_key, result_type),
        label,
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
        "Model": r["name"], "Strategy": r["strategy"],
        **{f"Base {METRIC_LABELS[m]}": round(r[f"base_{m}"], 4) for m in METRICS},
        **{f"Enh {METRIC_LABELS[m]}":  round(r[f"enh_{m}"],  4) for m in METRICS},
    } for r in rows]
    return dcc.send_data_frame(pd.DataFrame(records).to_csv, "model_leaderboard.csv", index=False)