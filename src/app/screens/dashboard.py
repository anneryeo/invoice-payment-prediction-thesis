from dash import html, dcc
from src.app.components.charts import LineChart
from src.app.components.tables import DataTableComponent

class DashboardScreen:
    def __init__(self, app):
        self.app = app

    def layout(self):
        return html.Div(
            className="dashboard-container",
            children=[
                html.H2("Main Dashboard", className="dashboard-title"),

                # KPI Cards
                html.Div(
                    className="kpi-container",
                    children=[
                        html.Div([
                            html.Div("120", className="kpi-value"),
                            html.Div("Invoices >60 Days", className="kpi-label")
                        ], className="kpi-card overdue"),

                        html.Div([
                            html.Div("80", className="kpi-value"),
                            html.Div("Invoices 31–60 Days", className="kpi-label")
                        ], className="kpi-card warning"),

                        html.Div([
                            html.Div("200", className="kpi-value"),
                            html.Div("Invoices 1–30 Days", className="kpi-label")
                        ], className="kpi-card safe"),

                        html.Div([
                            html.Div("92%", className="kpi-value"),
                            html.Div("Active Model Accuracy", className="kpi-label")
                        ], className="kpi-card info"),
                    ]
                ),

                # Charts
                html.Div([
                    LineChart("collections_chart", title="Collections Breakdown"),
                    LineChart("trend_chart", title="Overdue Invoices Trend"),
                ], className="chart-container"),

                # Drill-down Table
                html.Div([
                    html.H3("Invoices Predicted to be Late"),
                    DataTableComponent("invoice_table", ["Student ID", "Grade Level", "Amount", "Due Date", "Predicted Delay"]),
                    html.Button("Export CSV", id="dashboard-export-btn", className="primary-btn"),
                ], className="section"),

                # Model Insights
                html.Div([
                    html.H3("Model Insights"),
                    dcc.Dropdown(id="model_dropdown", placeholder="Select a model"),
                    dcc.Graph(id="auc_graph"),
                    dcc.Graph(id="feature_importance"),
                ], className="section"),

                # Data Management Buttons
                html.Div([
                    html.Button("Refresh Invoices", id="refresh_btn", className="primary-btn"),
                    html.Button("Retrain Model", id="retrain_btn", className="secondary-btn"),
                    # Navigates to Screen 2 — Model Analysis
                    # Wired in app/callbacks/dashboard_callbacks.py
                    html.Button("View Saved Models", id="view_models_btn", className="secondary-btn"),
                ], className="button-group"),

                # Audit Snapshot
                html.Div([
                    html.P("Last Data Upload: 2026-03-01 14:32"),
                    html.P("Last Training Run: 2026-03-02 09:15"),
                    html.P("Performed by: RJ"),
                ], className="audit-info")
            ]
        )
