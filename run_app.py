from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from src.app.utils.model_manager import has_trained_models
from src.app.components.navbar import Sidebar

from src.app import dash_app, server

# ── Callback registration (imports are side-effectful — order matters) ────────
import src.app.screens.initial_setup.intial_setup_layout                                         # registers step 1-5 + step-advancement callbacks
import src.app.screens.initial_setup.callbacks.step_4                                            # registers all dashboard callbacks (core, filters, screen_1)
import src.app.screens.model_analysis.callbacks.model_analysis_callbacks                         # registers screen_2 + navigation — must follow step_4
from src.app.screens.initial_setup.callbacks.initial_setup_layout_step_renderer import initial_setup_layout

# ── Screen classes ────────────────────────────────────────────────────────────
from src.app.screens.initial_setup.intial_setup_layout import InitialSetupScreen
from src.app.screens.model_analysis.layout import ModelAnalysisScreen
from src.app.screens.dashboard import DashboardScreen
from src.app.screens.invoice_drilldown import InvoiceDrilldownScreen
from src.app.screens.audit_logs import AuditLogsScreen
from src.app.screens.settings import SettingsScreen

# ── Instantiate screens ───────────────────────────────────────────────────────
initial_setup     = InitialSetupScreen(dash_app)
model_analysis    = ModelAnalysisScreen(dash_app)
dashboard         = DashboardScreen(dash_app)
invoice_drilldown = InvoiceDrilldownScreen(dash_app)
audit_logs        = AuditLogsScreen(dash_app)
settings_screen   = SettingsScreen(dash_app)

# ── Pre-load session on startup if models already exist ───────────────────────
# This means the MODELS dict is populated immediately rather than waiting for
# the analysis screen to mount its auto-load interval.
if has_trained_models():
    try:
        from src.app.screens.comparative_model_dashboard_template.utils.session_loader import (
            activate_session,
        )
        activate_session()
    except Exception as _e:
        print(f"[startup] Could not pre-load session: {_e}")

# ── Root layout ───────────────────────────────────────────────────────────────
# The setup layout (initial-setup-wrapper) is mounted PERMANENTLY so its
# stores/intervals/callback targets always exist in the DOM.
# The sidebar is rendered by the router — hidden on /setup routes.
# The app-shell uses flex: sidebar (fixed) + main-content (scrollable).
_SETUP_ROUTES = ("/", "/setup")

dash_app.layout = html.Div(
    id="root",
    children=[
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="finalization-complete", data=False),

        # Setup wizard — permanently in DOM, shown/hidden via display style
        html.Div(
            id="initial-setup-wrapper",
            children=[initial_setup_layout],
            style={"display": "none"},
        ),

        # App shell — sidebar + content (hidden during setup)
        html.Div(
            id="app-shell",
            className="",
            children=[
                html.Div(id="sidebar-container"),
                html.Div(
                    className="main-content",
                    children=[
                        html.Div(id="page-content", className="page-container"),
                    ],
                ),
            ],
            style={"display": "none"},
        ),
    ],
)


# ── Router ────────────────────────────────────────────────────────────────────
# Single callback owns ALL routing, redirect, sidebar, and shell visibility.
@dash_app.callback(
    Output("page-content",           "children"),
    Output("initial-setup-wrapper",  "style"),
    Output("app-shell",              "style"),
    Output("sidebar-container",      "children"),
    Output("url", "pathname",        allow_duplicate=True),
    Input("url", "pathname"),
    Input("finalization-complete",   "data"),
    prevent_initial_call="initial_duplicate",
)
def display_page(pathname, finalization_complete):
    setup_visible   = {"display": "block"}
    setup_hidden    = {"display": "none"}
    shell_visible   = {"display": "flex"}
    shell_hidden    = {"display": "none"}
    models_ready    = has_trained_models()

    # Post-finalization: training just completed → go to dashboard
    if finalization_complete:
        return (
            dashboard.layout(),
            setup_hidden,
            shell_visible,
            Sidebar("/dashboard"),
            "/dashboard",
        )

    # No models → always redirect to setup regardless of route
    if not models_ready:
        redirect = "/setup" if pathname not in _SETUP_ROUTES else no_update
        return None, setup_visible, shell_hidden, None, redirect

    # Models exist + on a setup route → redirect to dashboard
    if pathname in _SETUP_ROUTES:
        return (
            dashboard.layout(),
            setup_hidden,
            shell_visible,
            Sidebar("/dashboard"),
            "/dashboard",
        )

    # ── Named routes ──────────────────────────────────────────────────────────
    route_map = {
        "/dashboard": (dashboard.layout,         "/dashboard"),
        "/analysis":  (model_analysis.layout,    "/analysis"),
        "/drilldown": (invoice_drilldown.layout, "/drilldown"),
        "/logs":      (audit_logs.layout,        "/logs"),
        "/settings":  (settings_screen.layout,   "/settings"),
    }

    if pathname in route_map:
        layout_fn, active = route_map[pathname]
        return layout_fn(), setup_hidden, shell_visible, Sidebar(active), no_update

    # Fallback → dashboard
    return (
        dashboard.layout(),
        setup_hidden,
        shell_visible,
        Sidebar("/dashboard"),
        "/dashboard",
    )


if __name__ == "__main__":
    dash_app.run(debug=True, use_reloader=False)
