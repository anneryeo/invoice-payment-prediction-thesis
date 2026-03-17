import os

from dash import Input, Output, no_update

from app import dash_app
from ..constants import MODELS, set_class_labels
from ..utils.data_loaders import load_models_from_results
from utils.data_loaders.read_settings_json import read_settings_json
from machine_learning.utils.io.load_results_from_folder import SessionStore


# ── Session store ─────────────────────────────────────────────────────────────
_store = SessionStore(read_settings_json()["Training"]["RESULTS_ROOT"])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@dash_app.callback(
    Output("session-selector-dropdown", "options"),
    Output("session-selector-dropdown", "value"),
    Input("step4-data-loaded", "data"),
    prevent_initial_call=True,
)
def populate_session_dropdown(_loaded):
    """
    Fills the session dropdown with all dated folders on mount.
    Pre-selects the most recent session so the dashboard is never blank.
    """
    dirs = _store.list()
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
        db_path = _store.path(selected_folder)
    except FileNotFoundError as exc:
        print(f"[screen2] {exc}")
        return no_update

    try:
        session = _store.load(selected_folder)
        MODELS.clear()
        MODELS.update(load_models_from_results(db_path))
        set_class_labels(session["class_mappings"])
        print(f"[screen2] Loaded {len(MODELS)} models from {db_path}")
    except Exception as exc:
        print(f"[screen2] WARNING – could not load {db_path}: {exc}")
        MODELS.clear()

    return True