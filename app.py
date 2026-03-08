from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

from app.utils.model_manager import has_trained_models

from app import dash_app, server

# ── Callback registration (imports are side-effectful — order matters) ────────
import app.screens.initial_setup.intial_setup_layout                                         # registers step 1-5 + step-advancement callbacks
import app.screens.initial_setup.callbacks.step_4                                            # registers all dashboard callbacks
from app.screens.initial_setup.callbacks.initial_setup_layout_step_renderer import initial_setup_layout # registers session-selector callbacks (Screen 2)

# ── Screen classes ────────────────────────────────────────────────────────────
from app.screens.initial_setup.intial_setup_layout import InitialSetupScreen
from app.screens.model_analysis.layout import ModelAnalysisScreen # Standalone screen for comparing models
from app.screens.dashboard import DashboardScreen
from app.screens.invoice_drilldown import InvoiceDrilldownScreen
from app.screens.audit_logs import AuditLogsScreen
from app.screens.settings import SettingsScreen

# ── Instantiate screens ───────────────────────────────────────────────────────
initial_setup   = InitialSetupScreen(dash_app)
model_analysis  = ModelAnalysisScreen(dash_app)
dashboard       = DashboardScreen(dash_app)
invoice_drilldown = InvoiceDrilldownScreen(dash_app)
audit_logs      = AuditLogsScreen(dash_app)
settings        = SettingsScreen(dash_app)

# ── Root layout ───────────────────────────────────────────────────────────────
# initial_setup_layout is kept here so that its dcc.Stores and step-flow
# callbacks are always registered, regardless of which page is active.
# The visible step content is rendered inside it by render_step().
dash_app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content"),
    initial_setup_layout,   # persistent stores + hidden step runner
])


# ── Router ────────────────────────────────────────────────────────────────────
@dash_app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    if pathname == "/setup":
        return initial_setup.layout()
    elif pathname == "/analysis":
        return model_analysis.layout()
    elif pathname == "/dashboard":
        return dashboard.layout() if has_trained_models() else initial_setup.layout()
    elif pathname == "/drilldown":
        return invoice_drilldown.layout()
    elif pathname == "/logs":
        return audit_logs.layout()
    elif pathname == "/settings":
        return settings.layout()
    else:
        return dashboard.layout() if has_trained_models() else initial_setup.layout()


if __name__ == "__main__":
    dash_app.run(debug=True)