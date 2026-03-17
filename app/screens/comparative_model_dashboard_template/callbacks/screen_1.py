from dash import Input, Output, State, no_update

from app import dash_app
from app.screens.comparative_model_dashboard_template.constants import (
    MODELS,
    _model_display,
    _strategy_display,
    set_class_labels,
)
from app.screens.comparative_model_dashboard_template.utils.chart_builders import build_leaderboard_rows
from app.screens.comparative_model_dashboard_template.utils.data_loaders import load_models_from_results
from utils.data_loaders.read_settings_json import read_settings_json
from machine_learning.utils.io.load_results_from_folder import SessionStore


_store = SessionStore(read_settings_json()["Training"]["RESULTS_ROOT"])


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
        db_path = _store.path()
        session = _store.load()
        MODELS.update(load_models_from_results(db_path))
        set_class_labels(session["class_mappings"])
        print(f"[screen1] Loaded {len(MODELS)} models from {db_path}")
    except Exception as exc:
        print(f"[screen1] WARNING – could not load results.db: {exc}")
        MODELS.clear()
    return True


@dash_app.callback(
    Output("selected-model-store", "data", allow_duplicate=True),
    Input("step4-data-loaded",     "data"),
    State("selected-model-store",  "data"),
    prevent_initial_call=True,
)
def initialise_selection(loaded, current_selected):
    if not loaded or not MODELS:
        return no_update
    if current_selected:
        return no_update
    rows = build_leaderboard_rows("f1_macro", "enhanced", sort_result_type="enhanced")
    return rows[0]["key"] if rows else no_update


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
