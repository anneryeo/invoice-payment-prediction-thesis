from dash import Input, Output, State, no_update

from src.app import dash_app
from src.app.screens.comparative_model_dashboard_template.constants import MODELS
from src.app.screens.comparative_model_dashboard_template.utils.session_loader import (
    activate_session,
    get_store,
)


@dash_app.callback(
    Output("session-selector-dropdown", "options"),
    Output("session-selector-dropdown", "value"),
    Input("analysis-mount-interval",    "n_intervals"),
    prevent_initial_call=True,
)
def populate_session_dropdown(n_intervals):
    """
    Fires once when the analysis screen mounts (via one-shot interval).
    Fills the session dropdown and pre-selects the most recent session
    so the dashboard is never blank on arrival.
    """
    dirs = get_store().list()
    if not dirs:
        return [], None
    options = [{"label": d, "value": d} for d in dirs]
    return options, dirs[0]


@dash_app.callback(
    Output("step4-data-loaded", "data", allow_duplicate=True),
    Input("session-selector-dropdown", "value"),
    prevent_initial_call=True,
)
def load_selected_session(selected_folder):
    """
    Reloads MODELS from the chosen session's results.db.
    Returns True to step4-data-loaded so all downstream callbacks
    (leaderboard, charts, selection) re-run automatically.
    """
    if not selected_folder:
        return no_update
    try:
        n = activate_session(selected_folder, clear_first=True)
        print(f"[screen2] Loaded {n} models from session {selected_folder!r}")
    except Exception as exc:
        print(f"[screen2] WARNING – could not load session {selected_folder!r}: {exc}")
        MODELS.clear()
    return True
