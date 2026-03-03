from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

from MachineLearning.Utils.model_manager import has_trained_models

import app.callbacks.initial_setup_callbacks


# Import screen classes
from app.screens.initial_setup import InitialSetupScreen
from app.screens.model_training import ModelTrainingScreen
from app.screens.model_comparison import ModelComparisonScreen
from app.screens.dashboard import DashboardScreen
from app.screens.invoice_drilldown import InvoiceDrilldownScreen
from app.screens.audit_logs import AuditLogsScreen
from app.screens.settings import SettingsScreen

# Initialize Dash app
app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)
server = app.server

# Instantiate screens
initial_setup = InitialSetupScreen(app)
model_training = ModelTrainingScreen(app)
model_comparison = ModelComparisonScreen(app)
dashboard = DashboardScreen(app)
invoice_drilldown = InvoiceDrilldownScreen(app)
audit_logs = AuditLogsScreen(app)
settings = SettingsScreen(app)

# Layout with routing
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

# Callbacks for routing
@app.callback(
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
        # Only show dashboard if models exist
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
        # Default route: check if models exist
        if has_trained_models():
            return dashboard.layout()
        else:
            return initial_setup.layout()



if __name__ == "__main__":
    app.run(debug=True)