# app/callbacks/model_analysis_callbacks.py
#
# Screen 2 — Standalone Analysis.
# Imports shared callback modules (core + filters are already registered by
# Screen 1's import chain; guard against double-registration is handled by
# Python's module cache — each module only executes once per process).
# Adds the session-selector callbacks that are unique to this screen.

import app.screens.comparative_model_dashboard_template.callbacks.core     # no-op if already imported
import app.screens.comparative_model_dashboard_template.callbacks.filters  # no-op if already imported
import app.screens.comparative_model_dashboard_template.callbacks.screen_2 # session dropdown + on-demand load

from app.screens.comparative_model_dashboard_template.dashboard_layout import build_dashboard_layout

# Consumed by model_analysis.py → ModelAnalysisScreen.layout()
html_analysis = build_dashboard_layout(show_session_selector=True)
