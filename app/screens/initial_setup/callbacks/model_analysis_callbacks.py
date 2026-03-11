import os
import re

from dash import Input, Output, State, no_update, html
from app import dash_app

from ...comparative_model_dashboard_template.constants import MODELS
from ...comparative_model_dashboard_template.utils.data_loaders import load_models_from_results
from ...comparative_model_dashboard_template.dashboard_layout import build_dashboard_layout

from utils.data_loaders.read_settings_json import read_settings_json


# ── Resolve Results root from settings ───────────────────────────────────────
_settings   = read_settings_json()
_config     = _settings.get("Config", [{}])[0]
_RESULTS_ROOT = _config.get("RESULTS_ROOT", "Results")

_DATE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}$")


def _list_session_dirs() -> list[str]:
    """Return all dated session folder names under Results/, newest first."""
    try:
        dirs = sorted(
            [d for d in os.listdir(_RESULTS_ROOT) if _DATE_RE.match(d)],
            reverse=True,
        )
    except FileNotFoundError:
        dirs = []
    return dirs


def _session_db_path(folder: str) -> str:
    return os.path.join(_RESULTS_ROOT, folder, "results.db")


# ── Public layout object ──────────────────────────────────────────────────────
# Screen 2: standalone analysis with session picker
html_analysis = build_dashboard_layout(show_session_selector=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SCREEN 2 — STANDALONE ANALYSIS  (session-selector callbacks)
#  All leaderboard / chart / filter callbacks are shared and already registered
#  by initial_setup_step_4_callbacks.py.  This file only adds the two callbacks
#  that are unique to Screen 2: populating the dropdown and loading on change.
# ══════════════════════════════════════════════════════════════════════════════

@dash_app.callback(
    Output("session-selector-dropdown", "options"),
    Output("session-selector-dropdown", "value"),
    Input("step4-data-loaded", "data"),   # fires once the dashboard mounts
    prevent_initial_call=True,
)
def populate_session_dropdown(_loaded):
    """
    Populate the session dropdown with all dated folders under Results/.
    Pre-selects the most recent session (index 0).
    Only runs when the session-selector-wrap is present in the DOM,
    i.e. when Screen 2 is active.
    """
    dirs = _list_session_dirs()
    if not dirs:
        return [], None

    options = [{"label": d, "value": d} for d in dirs]
    return options, dirs[0]   # pre-select newest


@dash_app.callback(
    Output("step4-data-loaded", "data", allow_duplicate=True),
    Input("session-selector-dropdown", "value"),
    prevent_initial_call=True,
)
def load_selected_session(selected_folder):
    """
    When the user picks a session from the dropdown, reload MODELS from
    that session's results.db and signal all downstream callbacks to re-run
    by toggling step4-data-loaded back to False then True.
    """
    if not selected_folder:
        return no_update

    db_path = _session_db_path(selected_folder)
    if not os.path.exists(db_path):
        print(f"[analysis] results.db not found: {db_path}")
        return no_update

    try:
        MODELS.clear()
        MODELS.update(load_models_from_results(db_path))
        print(f"[analysis] Loaded {len(MODELS)} models from {db_path}")
    except Exception as exc:
        print(f"[analysis] WARNING – could not load {db_path}: {exc}")
        MODELS.clear()

    # Returning True re-triggers all callbacks that depend on step4-data-loaded
    return True