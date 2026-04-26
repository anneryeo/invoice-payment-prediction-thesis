from dash import Input, Output, State, no_update

from src.app import dash_app
from src.app.screens.comparative_model_dashboard_template.constants import (
    MODELS,
    _model_display,
    _strategy_display,
)
from src.app.screens.comparative_model_dashboard_template.utils.chart_builders import build_leaderboard_rows
from src.app.screens.comparative_model_dashboard_template.utils.session_loader import activate_session


@dash_app.callback(
    Output("step4-data-loaded", "data"),
    Input("current_step",       "data"),
    State("step4-data-loaded",  "data"),
    prevent_initial_call=True,
)
def load_step4_data(current_step, already_loaded):
    if current_step != "progress-4":
        return no_update
    if already_loaded:
        return no_update
    try:
        n = activate_session()
        print(f"[screen1] Loaded {n} models")
    except Exception as exc:
        print(f"[screen1] WARNING – could not load results.db: {exc}")
        MODELS.clear()
    return True


# initialise_selection moved to core.py


@dash_app.callback(
    Output("confirm-fab-wrap",    "className"),
    Output("confirm-fab-label",   "children"),
    Input("selected-model-store", "data"),
    prevent_initial_call=True,
)
def toggle_confirm_fab(model_key):
    if not model_key or model_key not in MODELS:
        return "confirm-bar-wrap confirm-fab-hidden", ""

    data     = MODELS[model_key]
    name     = _model_display(data["model"])
    strategy = _strategy_display(data["balance_strategy"])

    return "confirm-bar-wrap confirm-fab-visible", f"{name}  ·  {strategy}"
