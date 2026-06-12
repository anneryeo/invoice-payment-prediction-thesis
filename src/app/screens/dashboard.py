# screens/dashboard.py
#
# KPI Dashboard — reads live data from the MODELS dict (populated by
# activate_session on startup) and the deployed model metadata.
# Falls back gracefully when data is unavailable.

from __future__ import annotations

import os
import pickle
import glob

import plotly.graph_objects as go
from dash import html, dcc, Input, Output, dash_table

from src.app import dash_app
from src.utils.data_loaders.read_settings_json import read_settings_json

# Payment bracket display config
_BRACKET_CONFIG = {
    "on_time": {"label": "On Time",    "color": "#2d6a4f"},
    "30_days": {"label": "1–30 Days",  "color": "#d97706"},
    "60_days": {"label": "31–60 Days", "color": "#c2410c"},
    "90_days": {"label": "61+ Days",   "color": "#b91c1c"},
}
_BRACKET_ORDER = ["on_time", "30_days", "60_days", "90_days"]


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_models_summary() -> dict:
    try:
        from src.app.screens.comparative_model_dashboard_template.constants import MODELS
        if not MODELS:
            from src.app.screens.comparative_model_dashboard_template.utils.session_loader import (
                activate_session,
            )
            activate_session()
        from src.app.screens.comparative_model_dashboard_template.constants import MODELS
        return MODELS
    except Exception:
        return {}


def _get_deployed_model_name() -> str:
    try:
        settings = read_settings_json()
        deployed_dir = settings.get("Training", {}).get("DEPLOYED_MODELS", "")
        candidates = [
            p for p in glob.glob(os.path.join(deployed_dir, "finalized_*.pkl"))
            if not p.endswith("finalized_survival_model.pkl")
        ]
        if candidates:
            name = os.path.basename(candidates[0])
            return name.replace("finalized_", "").replace(".pkl", "").replace("_", " ").title()
    except Exception:
        pass
    return "—"


def _load_class_distribution() -> dict | None:
    try:
        settings  = read_settings_json()
        cache_path = os.path.join(
            settings.get("Training", {}).get("RESULTS_ROOT", ""),
            "credit_sales_cache.pkl",
        )
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, "rb") as fh:
            import pandas as _pd
            df = _pd.read_pickle(fh)
        target = settings.get("Training", {}).get("target_feature", "dtp_bracket")
        if target not in df.columns:
            return None
        return df[target].value_counts().to_dict()
    except Exception:
        return None


def _best_model_stats(models: dict) -> dict:
    if not models:
        return {}
    best_key = max(
        models,
        key=lambda k: (models[k].get("evaluation", {})
                                .get("metrics", {})
                                .get("enhanced", {})
                                .get("f1_macro", 0.0)),
    )
    data    = models[best_key]
    metrics = data.get("evaluation", {}).get("metrics", {}).get("enhanced", {})
    return {
        "name":     data.get("model", best_key).replace("_", " ").title(),
        "strategy": data.get("balance_strategy", "").replace("_", " ").title(),
        "f1":       metrics.get("f1_macro", 0.0),
        "auc":      metrics.get("roc_auc_macro", 0.0),
    }


# ── Component builders ────────────────────────────────────────────────────────

def _kpi_card(label: str, value: str, meta: str = "", top_cls: str = "") -> html.Div:
    return html.Div(
        className=f"kpi-card {top_cls}",
        children=[
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(meta,  className="kpi-meta") if meta else None,
        ],
    )


def _bracket_bar(dist: dict) -> dcc.Graph:
    labels = [_BRACKET_CONFIG[k]["label"] for k in _BRACKET_ORDER if k in dist]
    values = [dist[k] for k in _BRACKET_ORDER if k in dist]
    colors = [_BRACKET_CONFIG[k]["color"] for k in _BRACKET_ORDER if k in dist]
    total  = sum(values) or 1

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:,}  ({v/total:.1%})" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:,} invoices<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=0, r=90, t=8, b=8),
        xaxis=dict(showgrid=True, gridcolor="#f0ede6", zeroline=False,
                   title="Invoice Count", title_font_size=11),
        yaxis=dict(showgrid=False, autorange="reversed"),
        font=dict(family="-apple-system, 'Segoe UI', sans-serif", size=12, color="#2b2b2b"),
        height=230,
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def _top_models_tbl(models: dict, n: int = 8) -> dash_table.DataTable:
    rows = []
    for key, data in models.items():
        m = data.get("evaluation", {}).get("metrics", {}).get("enhanced", {})
        rows.append({
            "Model":     data.get("model", key).replace("_", " ").title(),
            "Strategy":  data.get("balance_strategy", "").replace("_", " ").title(),
            "F1 ↑":      round(m.get("f1_macro", 0), 4),
            "AUC":       round(m.get("roc_auc_macro", 0), 4),
            "Accuracy":  round(m.get("accuracy", 0), 4),
            "_sort":     m.get("f1_macro", 0),
        })
    rows.sort(key=lambda r: r["_sort"], reverse=True)
    rows = [{k: v for k, v in r.items() if k != "_sort"} for r in rows[:n]]
    cols = [{"name": c, "id": c} for c in ("Model", "Strategy", "F1 ↑", "AUC", "Accuracy")]

    return dash_table.DataTable(
        data=rows, columns=cols,
        page_size=n,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#1b3a6b", "color": "white",
            "fontWeight": "600", "fontSize": "11px",
            "textTransform": "uppercase", "letterSpacing": "0.05em",
            "padding": "10px 14px", "border": "none",
        },
        style_cell={
            "textAlign": "left", "padding": "9px 14px",
            "fontSize": "13px", "color": "#2b2b2b",
            "border": "1px solid #e5e1d8",
            "fontFamily": "-apple-system, 'Segoe UI', sans-serif",
        },
        style_data_conditional=[
            {"if": {"row_index": 0}, "backgroundColor": "#fdf6e3", "fontWeight": "600"},
            {"if": {"row_index": "odd"}, "backgroundColor": "#fafaf8"},
        ],
    )


def _empty(icon: str, title: str, body: str = "") -> html.Div:
    return html.Div(className="empty-state", children=[
        html.Span(icon,  className="empty-state-icon"),
        html.P(title,    className="empty-state-title"),
        html.P(body,     className="empty-state-text") if body else None,
    ])


# ── Screen class ──────────────────────────────────────────────────────────────

class DashboardScreen:
    def __init__(self, app):
        self.app = app
        self._register_callbacks()

    def layout(self) -> html.Div:
        return html.Div(
            id="dashboard-root",
            children=[
                dcc.Interval(
                    id="dashboard-mount-interval",
                    interval=250, n_intervals=0, max_intervals=1,
                ),
                html.Div(className="page-header", children=[
                    html.H2("Dashboard", className="page-title"),
                    html.P(
                        "Model performance overview and invoice payment distribution.",
                        className="page-subtitle",
                    ),
                ]),
                # KPI row
                html.Div(id="dash-kpi-row",  className="kpi-grid"),
                # Charts
                html.Div(className="section-row cols-1-2", children=[
                    html.Div(className="card", children=[
                        html.Div(className="section-header", children=[
                            html.H4("Payment Bracket Distribution", className="section-title"),
                        ]),
                        html.Div(id="dash-dist-chart", className="card-body",
                                 children=[html.Div("Loading…", className="loading-state")]),
                    ]),
                    html.Div(className="card", children=[
                        html.Div(className="section-header", children=[
                            html.H4("Top Models by F1 Score (Enhanced)", className="section-title"),
                        ]),
                        html.Div(id="dash-top-models", style={"padding": "0"},
                                 children=[html.Div("Loading…", className="loading-state")]),
                    ]),
                ]),
            ],
        )

    def _register_callbacks(self):
        @dash_app.callback(
            Output("dash-kpi-row",    "children"),
            Output("dash-dist-chart", "children"),
            Output("dash-top-models", "children"),
            Input("dashboard-mount-interval", "n_intervals"),
            prevent_initial_call=True,
        )
        def _load(_n):
            models        = _load_models_summary()
            best          = _best_model_stats(models)
            dist          = _load_class_distribution()
            deployed_name = _get_deployed_model_name()

            # KPIs
            if best:
                kpi_row = [
                    _kpi_card("Best Model F1",      f"{best['f1']:.4f}",
                              best["name"], "accent-top"),
                    _kpi_card("Best Model AUC",     f"{best['auc']:.4f}",
                              f"Strategy: {best['strategy']}", "success-top"),
                    _kpi_card("Total Experiments",  f"{len(models):,}",
                              "Configurations benchmarked"),
                    _kpi_card("Deployed Model",     deployed_name,
                              "Currently active", "accent-top"),
                ]
            else:
                kpi_row = [html.Div(
                    className="card card-body",
                    style={"gridColumn": "1 / -1"},
                    children=[_empty(
                        "📂", "No results found.",
                        "Run the Setup Wizard to train and benchmark models.",
                    )],
                )]

            # Distribution chart
            dist_content = (
                [_bracket_bar(dist),
                 html.Div(f"Total: {sum(dist.values()):,} invoice records",
                          className="text-xs text-muted mt-1")]
                if dist
                else [_empty("📊", "No dataset available.",
                             "Complete the Setup Wizard to load invoice data.")]
            )

            # Top models table
            top_content = (
                [_top_models_tbl(models)]
                if models
                else [_empty("🔬", "No model results loaded.")]
            )

            return kpi_row, dist_content, top_content
