from dash import html, dcc

from src.app.screens.model_analysis.callbacks.model_analysis_callbacks import html_analysis


class ModelAnalysisScreen:
    """
    Screen 2 — Standalone Analysis.

    Wraps the shared dashboard layout with a full-page shell (no progress
    stepper, no confirm button).  The session selector is embedded inside
    html_analysis itself via build_dashboard_layout(show_session_selector=True).
    """

    def __init__(self, dash_app):
        self.dash_app = dash_app

    def layout(self) -> html.Div:
        return html.Div(
            className="analysis-container",
            children=[
                # Scoped stylesheet for this screen
                html.Link(rel="stylesheet", href="/assets/analysis.css"),

                # One-shot interval — fires once on mount, triggers
                # populate_session_dropdown in screen_2.py, then disables itself.
                dcc.Interval(
                    id="analysis-mount-interval",
                    interval=100,
                    n_intervals=0,
                    max_intervals=1,
                ),

                html.Div(
                    className="analysis-header",
                    children=[
                        html.H2("Model Analysis", className="analysis-title"),
                        html.Hr(className="analysis-divider"),
                    ],
                ),
                html_analysis,
            ],
        )
