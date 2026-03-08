import os
import re

from dash import Input, Output, no_update

from app import dash_app
from ..constants import MODELS
from ..utils.data_loaders import load_models_from_results

from utils.io.read_settings_json import read_settings_json


# ── Resolve Results root ──────────────────────────────────────────────────────
_settings     = read_settings_json()
_config       = _settings.get("Config", [{}])[0]
_RESULTS_ROOT = _config.get("RESULTS_ROOT", "Results")
_DATE_RE      = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}$")


def _list_session_dirs() -> list[str]:
    """All dated session folder names under Results/, newest first."""
    try:
        return sorted(
            [d for d in os.listdir(_RESULTS_ROOT) if _DATE_RE.match(d)],
            reverse=True,
        )
    except FileNotFoundError:
        return []


def _session_db_path(folder: str) -> str:
    return os.path.join(_RESULTS_ROOT, folder, "results.db")


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
    dirs = _list_session_dirs()
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

    db_path = _session_db_path(selected_folder)
    if not os.path.exists(db_path):
        print(f"[screen2] results.db not found: {db_path}")
        return no_update

    try:
        MODELS.clear()
        MODELS.update(load_models_from_results(db_path))
        print(f"[screen2] Loaded {len(MODELS)} models from {db_path}")
    except Exception as exc:
        print(f"[screen2] WARNING – could not load {db_path}: {exc}")
        MODELS.clear()

    return True
