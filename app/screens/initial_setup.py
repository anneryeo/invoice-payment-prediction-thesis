# screens/initial_setup.py
from dash import html, dcc

class InitialSetupScreen:
    def __init__(self, dash_app):
        self.dash_app = dash_app

    def layout(self):
        return html.Div(
            className="setup-container",
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

                # Step Content (default = Step 1 upload boxes with icons)
                html.Div(
                    id="step-content",
                    className="setup-step",
                    children=[
                        html.H3("Step 1: Upload your datasets", className="step-header"),

                        # Revenue upload
                        dcc.Upload(
                            id="upload_revenue",
                            children=html.Div([
                                html.Img(src="/assets/icons/csv_icon.png", className="upload-icon"),
                                html.Div("Drag & Drop or Click to Upload Revenue Ledger CSV")
                            ], className="upload-content"),
                            multiple=False,
                            className="upload-box"
                        ),
                        html.Div(id="upload_revenue_output", className="upload-output"),

                        # Enrollees upload
                        dcc.Upload(
                            id="upload_enrollees",
                            children=html.Div([
                                html.Img(src="/assets/icons/csv_icon.png", className="upload-icon"),
                                html.Div("Drag & Drop or Click to Upload Enrollee Information CSV")
                            ], className="upload-content"),
                            multiple=False,
                            className="upload-box"
                        ),
                        html.Div(id="upload_enrollees_output", className="upload-output"),

                        # Action buttons
                        html.Button("Confirm Uploads", id="upload_confirm_btn", className="setup-btn"),
                        html.Button("Next", id="next_btn", className="setup-btn", disabled=True)
                    ]
                )
            ]
        )