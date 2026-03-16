# app/callbacks/initial_setup_step_4_callbacks.py
#
# Screen 1 — Initial Setup, Step 4.
# This file is intentionally thin: it imports the shared callback modules to
# trigger their registration with Dash, then exposes html_step_4 for the
# step-render callback in initial_setup_callbacks.py.

import app.screens.comparative_model_dashboard_template.callbacks.core     # leaderboard, charts, sort, pagination, CSV
import app.screens.comparative_model_dashboard_template.callbacks.filters  # filter panel + stores
import app.screens.comparative_model_dashboard_template.callbacks.screen_1  # auto-load latest session + default selection

from app.screens.comparative_model_dashboard_template.dashboard_layout import build_dashboard_layout
from dash import Input, Output, State, html, no_update, callback_context

from app import dash_app
from app.screens.comparative_model_dashboard_template.constants import (
    MODELS, METRICS, METRIC_LABELS,
    _model_display, _strategy_display,
)

# Consumed by initial_setup_layout_step_renderer → placed into step4-content.
html_step_4 = build_dashboard_layout(
    show_session_selector=False,
    show_confirm_button=True,
)


# ── Open / close modal ────────────────────────────────────────────────────────
@dash_app.callback(
    Output("model-summary-modal",    "className"),
    Output("modal-model-name",       "children"),
    Output("modal-strategy-pill",    "children"),
    Output("modal-params-content",   "children"),
    Output("modal-baseline-metrics", "children"),
    Output("modal-enhanced-metrics", "children"),
    Output("selected-model-data",    "data"),
    Output("current_step", "data", allow_duplicate=True),
    Input("confirm-model-btn",  "n_clicks"),
    Input("modal-close-btn",    "n_clicks"),
    Input("modal-cancel-btn",   "n_clicks"),
    Input("modal-proceed-btn",  "n_clicks"),
    State("selected-model-store", "data"),
    prevent_initial_call=True,
)
def toggle_modal(confirm_clicks, close_clicks, cancel_clicks, proceed_clicks, model_key):
    ctx = callback_context
    if not ctx.triggered:
        return (no_update,) * 8

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger in ("modal-close-btn", "modal-cancel-btn"):
        return "modal-overlay modal-hidden", no_update, no_update, no_update, no_update, no_update, no_update, no_update

    if trigger == "modal-proceed-btn":
        # Safe to write "modal-overlay modal-hidden" here because model-summary-modal
        # lives in step4-content which is permanently in the DOM — it is never
        # unmounted, so this write cannot race with a node deletion.
        return "modal-overlay modal-hidden", no_update, no_update, no_update, no_update, no_update, no_update, "progress-5"

    # confirm-model-btn → open modal
    if not model_key or model_key not in MODELS:
        return "modal-overlay modal-hidden", no_update, no_update, no_update, no_update, no_update, no_update, no_update

    data = MODELS[model_key]

    model_name     = _model_display(data["model"])
    strategy_label = _strategy_display(data["balance_strategy"])

    params = data.get("parameters", {})
    params_content = html.Div([
        html.Div(className="modal-param-row", children=[
            html.Span(k.replace("_", " ").title(), className="modal-param-key"),
            html.Span(str(v),                      className="modal-param-val"),
        ])
        for k, v in sorted(params.items())
    ]) if params else html.Span("Default parameters", className="modal-param-empty")

    def _metric_rows(result_type: str):
        m = data[result_type]["evaluation"]["metrics"]
        return html.Div([
            html.Div(className="modal-metric-row", children=[
                html.Span(METRIC_LABELS[k], className="modal-metric-label"),
                html.Span(f"{m.get(k, 0):.4f}", className="modal-metric-val"),
            ])
            for k in METRICS
        ])

    snapshot = {
        "key":      model_key,
        "model":    data["model"],
        "name":     model_name,
        "strategy": data["balance_strategy"],
        "parameters": params,
        "baseline": data["baseline"]["evaluation"]["metrics"],
        "enhanced": data["enhanced"]["evaluation"]["metrics"],
    }

    return (
        "modal-overlay modal-visible",
        model_name,
        strategy_label,
        params_content,
        _metric_rows("baseline"),
        _metric_rows("enhanced"),
        snapshot,
        no_update,  # current_step unchanged when opening the modal
    )