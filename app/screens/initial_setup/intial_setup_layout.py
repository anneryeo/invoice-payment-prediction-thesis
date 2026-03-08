# screens/initial_setup.py
from dash import html, dcc

class InitialSetupScreen:
    def __init__(self, dash_app):
        self.dash_app = dash_app

    def layout(self):
        return html.Div(
            className="setup-container",
            children=[
                # Hidden store to track current step
                dcc.Store(id="current_step", data="progress-1"),
                dcc.Store(id="stored_revenue"),
                dcc.Store(id="stored_enrollees"),

                # Sticky top — title + progress bar stay visible while scrolling
                html.Div(
                    className="setup-sticky-top",
                    children=[
                        # Top Title Section
                        html.Div(
                            className="setup-header",
                            children=[
                                html.H2("Initial Setup", className="setup-title"),
                                html.Hr(className="setup-divider")
                            ]
                        ),

                        # Progress Header
                        html.Div(
                            className="progress-header",
                            children=[
                                html.Div([
                                    html.Div("1", className="step-number"),
                                    html.Span("Upload Dataset", className="step-label")
                                ], id="progress-1", className="progress-step active"),

                                html.Div([
                                    html.Div("2", className="step-number"),
                                    html.Span("Model Training Selection", className="step-label")
                                ], id="progress-2", className="progress-step future"),

                                html.Div([
                                    html.Div("3", className="step-number"),
                                    html.Span("Waiting for Model Results", className="step-label")
                                ], id="progress-3", className="progress-step future"),

                                html.Div([
                                    html.Div("4", className="step-number"),
                                    html.Span("Model Result Analysis", className="step-label")
                                ], id="progress-4", className="progress-step future"),

                                html.Div([
                                    html.Div("5", className="step-number"),
                                    html.Span("Finalization", className="step-label")
                                ], id="progress-5", className="progress-step future"),
                            ]
                        ),
                    ]
                ),

                # Step Content (populated dynamically by callback)
                html.Div(id="step-content", className="setup-step")
            ]
        )