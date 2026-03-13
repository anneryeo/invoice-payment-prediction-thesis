from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from app.utils.model_manager import has_trained_models

from app import dash_app, server

# ── Callback registration (imports are side-effectful — order matters) ────────
import app.screens.initial_setup.intial_setup_layout                                         # registers step 1-5 + step-advancement callbacks
import app.screens.initial_setup.callbacks.step_4                                            # registers all dashboard callbacks
from app.screens.initial_setup.callbacks.initial_setup_layout_step_renderer import initial_setup_layout

# ── Screen classes ────────────────────────────────────────────────────────────
from app.screens.initial_setup.intial_setup_layout import InitialSetupScreen
from app.screens.model_analysis.layout import ModelAnalysisScreen
from app.screens.dashboard import DashboardScreen
from app.screens.invoice_drilldown import InvoiceDrilldownScreen
from app.screens.audit_logs import AuditLogsScreen
from app.screens.settings import SettingsScreen

# ── Instantiate screens ───────────────────────────────────────────────────────
initial_setup     = InitialSetupScreen(dash_app)
model_analysis    = ModelAnalysisScreen(dash_app)
dashboard         = DashboardScreen(dash_app)
invoice_drilldown = InvoiceDrilldownScreen(dash_app)
audit_logs        = AuditLogsScreen(dash_app)
settings          = SettingsScreen(dash_app)

# ── Root layout ───────────────────────────────────────────────────────────────
# initial_setup_layout is mounted PERMANENTLY so its stores, intervals, and
# callback targets always exist in the DOM regardless of which page is active.
# page-content is used ONLY for non-setup screens (analysis, dashboard, etc.).
# When on /setup the setup layout is already visible; page-content is empty.
_SETUP_ROUTES = ("/", "/setup")

dash_app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="finalization-complete", data=False),
    # Hidden by default — prevents flash of setup UI before callback fires on non-setup routes
    html.Div(id="initial-setup-wrapper", children=[initial_setup_layout],
             style={"display": "none"}),
    html.Div(id="page-content"),
])


# ── Router ────────────────────────────────────────────────────────────────────
# Single callback owns ALL routing + redirect logic to avoid race conditions.
# Handles: normal navigation, startup redirect (models exist), post-finalization redirect.
@dash_app.callback(
    Output("page-content",          "children"),
    Output("initial-setup-wrapper", "style"),
    Output("url", "pathname",       allow_duplicate=True),
    Input("url", "pathname"),
    Input("finalization-complete",  "data"),
    prevent_initial_call="initial_duplicate",
)
def display_page(pathname, finalization_complete):
    setup_visible = {"display": "block"}
    setup_hidden  = {"display": "none"}
    models_ready  = has_trained_models()

    # Post-finalization: training just completed -> go to dashboard
    if finalization_complete:
        return dashboard.layout(), setup_hidden, "/dashboard"

    # No models -> always redirect to setup regardless of route
    if not models_ready:
        redirect = "/setup" if pathname not in _SETUP_ROUTES else no_update
        return None, setup_visible, redirect

    # Models exist + on a setup route -> redirect to dashboard
    if pathname in _SETUP_ROUTES:
        return dashboard.layout(), setup_hidden, "/dashboard"

    # Models exist + named routes
    if pathname == "/dashboard":
        return dashboard.layout(), setup_hidden, no_update
    elif pathname == "/analysis":
        return model_analysis.layout(), setup_hidden, no_update
    elif pathname == "/drilldown":
        return invoice_drilldown.layout(), setup_hidden, no_update
    elif pathname == "/logs":
        return audit_logs.layout(), setup_hidden, no_update
    elif pathname == "/settings":
        return settings.layout(), setup_hidden, no_update
    else:
        return dashboard.layout(), setup_hidden, "/dashboard"


if __name__ == "__main__":
    dash_app.run(debug=True, use_reloader=False)
