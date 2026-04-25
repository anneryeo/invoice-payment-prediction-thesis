# app/callbacks/model_analysis_callbacks.py
#
# Screen 2 — Standalone Analysis.
# Imports shared callback modules (core + filters are already registered by
# Screen 1's import chain; guard against double-registration is handled by
# Python's module cache — each module only executes once per process).
# Adds the session-selector callbacks that are unique to this screen.
# Also handles navigation from the main dashboard's "View Saved Models" button.

import src.app.screens.comparative_model_dashboard_template.callbacks.core     # no-op if already imported
import src.app.screens.comparative_model_dashboard_template.callbacks.filters  # no-op if already imported
import src.app.screens.comparative_model_dashboard_template.callbacks.screen_2 # session dropdown + on-demand load

from src.app.screens.comparative_model_dashboard_template.dashboard_layout import build_dashboard_layout
from dash import Input, Output, no_update
from dash import dcc

from src.app import dash_app

# Consumed by model_analysis.py → ModelAnalysisScreen.layout()
html_analysis = build_dashboard_layout(show_session_selector=True)


# ── Navigate to Model Analysis (Screen 2) ─────────────────────────────────────

@dash_app.callback(
    Output("url", "pathname"),
    Input("view_models_btn", "n_clicks"),
    prevent_initial_call=True,
)
def navigate_to_model_analysis(n_clicks):
    """
    Fires when the user clicks "View Saved Models" on the main dashboard.
    Updates the URL to /analysis, which the router in app.py maps to
    ModelAnalysisScreen.
    """
    if not n_clicks:
        return no_update
    return "/analysis"
