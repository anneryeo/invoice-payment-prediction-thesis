# app.py
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

from MachineLearning.Utils.model_manager import has_trained_models

# Import the single Dash instance created in app/__init__.py
from app import dash_app, server

# Import callbacks (they attach automatically to the app instance)
import app.callbacks.initial_setup_callbacks

# Import screen classes
from app.screens.initial_setup import InitialSetupScreen
from app.screens.model_training import ModelTrainingScreen
from app.screens.model_comparison import ModelComparisonScreen
from app.screens.dashboard import DashboardScreen
from app.screens.invoice_drilldown import InvoiceDrilldownScreen
from app.screens.audit_logs import AuditLogsScreen
from app.screens.settings import SettingsScreen

# Instantiate screens
initial_setup = InitialSetupScreen(app)
model_training = ModelTrainingScreen(app)
model_comparison = ModelComparisonScreen(app)
dashboard = DashboardScreen(app)
invoice_drilldown = InvoiceDrilldownScreen(app)
audit_logs = AuditLogsScreen(app)
settings = SettingsScreen(app)

# Layout with routing
dash_app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

# Routing callback
@dash_app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/setup":
        return initial_setup.layout()
    elif pathname == "/train":
        return model_training.layout()
    elif pathname == "/compare":
        return model_comparison.layout()
    elif pathname == "/dashboard":
        if has_trained_models():
            return dashboard.layout()
        else:
            return initial_setup.layout()
    elif pathname == "/drilldown":
        return invoice_drilldown.layout()
    elif pathname == "/logs":
        return audit_logs.layout()
    elif pathname == "/settings":
        return settings.layout()
    else:
        if has_trained_models():
            return dashboard.layout()
        else:
            return initial_setup.layout()

if __name__ == "__main__":
    dash_app.run(debug=True)