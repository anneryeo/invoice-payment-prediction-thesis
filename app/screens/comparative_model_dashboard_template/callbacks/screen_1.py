from dash import Input, Output, State, no_update

from app import dash_app
from app.screens.comparative_model_dashboard_template.constants import MODELS
from ..utils.chart_builders import build_leaderboard_rows
from ..utils.data_loaders import load_models_from_results
from utils.io.read_settings_json import read_settings_json
from utils.io.latest_results_path import get_latest_results_path


@dash_app.callback(
    Output("step4-data-loaded", "data"),
    Input("step4-data-loaded", "data"),
    prevent_initial_call=False,
)
def load_step4_data(already_loaded):
    """
    Fires once when the dashboard first mounts.
    Loads the latest results.db automatically — no user action required.
    Skipped on subsequent triggers because already_loaded is True.
    """
    if already_loaded:
        return True

    try:
        latest_results_path = _get_latest_results_path_from_settings()
        MODELS.update(load_models_from_results(latest_results_path))
        print(f"[screen1] Loaded {len(MODELS)} models from {latest_results_path}")
    except Exception as exc:
        print(f"[screen1] WARNING – could not load results.db: {exc}")
        MODELS.clear()

    return True


@dash_app.callback(
    Output("selected-model-store", "data", allow_duplicate=True),
    Input("step4-data-loaded", "data"),
    State("selected-model-store", "data"),
    prevent_initial_call=True,
)
def initialise_selection(loaded, current_selected):
    """Auto-select the rank-#1 model once data is loaded."""
    if not loaded or not MODELS:
        return no_update
    if current_selected:
        return no_update
    rows = build_leaderboard_rows("f1_macro", "enhanced", sort_result_type="enhanced")
    return rows[0]["key"] if rows else no_update


def _get_latest_results_path_from_settings():
    settings_json = read_settings_json()
    config = settings_json.get("Config", [{}])[0]
    results_root = config.get("RESULTS_ROOT")
    latest_results_path = get_latest_results_path(results_root)

    return latest_results_path