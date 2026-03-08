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

# Consumed by initial_setup_callbacks.py → render_step()
html_step_4 = build_dashboard_layout(show_session_selector=False)
