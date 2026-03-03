from dash import Input, Output, html, dcc, ctx, State, no_update
from app import dash_app

@dash_app.callback(
    Output("step-content", "children"),
    [
        Input("progress-1", "n_clicks"),
        Input("progress-2", "n_clicks"),
        Input("progress-3", "n_clicks"),
        Input("progress-4", "n_clicks"),
        Input("progress-5", "n_clicks"),
    ],
    prevent_initial_call=False   # run at startup
)
def render_step(step1, step2, step3, step4, step5):
    # Default to step 1 if nothing triggered yet
    step = ctx.triggered_id if ctx.triggered_id is not None else "progress-1"

    if step == "progress-1":
        return html.Div([
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
        ])

    elif step == "progress-2":
        return html.Div([
            html.Div("2", className="step-number"),
            html.H3("Model Training Selection", className="step-header"),
            dcc.Checklist(
                id="model_selection",
                options=[
                    {"label": "Random Forest", "value": "rf"},
                    {"label": "XGBoost", "value": "xgb"},
                ]
            ),
            html.Button("Confirm Selection", id="confirm_btn")
        ])

    elif step == "progress-3":
        return html.Div([
            html.Div("3", className="step-number"),
            html.H3("Waiting for Model Results", className="step-header"),
            html.Div("Training in progress...")
        ])

    elif step == "progress-4":
        return html.Div([
            html.Div("4", className="step-number"),
            html.H3("Model Result Analysis", className="step-header"),
            dcc.Dropdown(id="model_summary_dropdown", placeholder="Select a model"),
            dcc.Graph(id="auc_graph")
        ])

    elif step == "progress-5":
        return html.Div([
            html.Div("5", className="step-number"),
            html.H3("Finalization", className="step-header"),
            html.Button("Finalize Model", id="finalize_btn")
        ])


# Revenue upload: hide box, show filename + X
@dash_app.callback(
    [
        Output("upload_revenue", "style"),
        Output("upload_revenue_output", "children"),
    ],
    Input("upload_revenue", "contents"),
    State("upload_revenue", "filename"),
    prevent_initial_call=True
)
def show_revenue_filename(contents, filename):
    if contents and filename:
        return (
            {"display": "none"},
            html.Div(
                className="file-display",
                children=[
                    html.Img(src="/assets/icons/csv_icon.png", className="file-icon"),
                    html.Span(filename, className="file-name"),
                    html.Button("X", id="delete_revenue", className="delete-btn")
                ]
            )
        )
    return {"display": "block"}, None

# Enrollees upload: hide box, show filename + X
@dash_app.callback(
    [Output("upload_enrollees", "style"),
     Output("upload_enrollees_output", "children")],
    Input("upload_enrollees", "contents"),
    State("upload_enrollees", "filename"),
    prevent_initial_call=True
)
def show_enrollees_filename(contents, filename):
    if contents and filename:
        return (
            {"display": "none"},
            html.Div(
                className="file-display",
                children=[
                    html.Img(src="/assets/icons/csv_icon.png", className="file-icon"),
                    html.Span(filename, className="file-name"),
                    html.Button("X", id="delete_enrollees", className="delete-btn")
                ]
            )
        )
    return {"display": "block"}, None


# Clear Revenue file when X is clicked
@dash_app.callback(
    [
        Output("upload_revenue", "style", allow_duplicate=True),
        Output("upload_revenue_output", "children", allow_duplicate=True),
        Output("upload_revenue", "contents", allow_duplicate=True),
    ],
    Input("delete_revenue", "n_clicks"),
    prevent_initial_call=True
)
def clear_revenue_file(n_clicks):
    if n_clicks:
        # Reset style, clear children, and reset contents
        return {"display": "block"}, None, None
    return no_update, no_update, no_update


# Clear Enrollees file when X is clicked
@dash_app.callback(
    [
        Output("upload_enrollees", "style", allow_duplicate=True),
        Output("upload_enrollees_output", "children", allow_duplicate=True),
        Output("upload_enrollees", "contents", allow_duplicate=True),
    ],
    Input("delete_enrollees", "n_clicks"),
    prevent_initial_call=True
)
def clear_enrollees_file(n_clicks):
    if n_clicks:
        return {"display": "block"}, None, None
    return no_update, no_update, no_update