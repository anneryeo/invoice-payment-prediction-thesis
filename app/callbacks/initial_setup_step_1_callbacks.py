from dash import Input, Output, State, html, dcc, no_update

from app import dash_app

html_step_1 = (
    html.Div([
        html.H3("Step 1: Upload your datasets", className="step-header"),

        # Upload box for the revenues ledger
        dcc.Upload(
            id="upload_revenues",
            children=html.Div([
                html.Img(src="/assets/icons/xlsx_icon.png", className="upload-icon"),
                html.Div("Drag & Drop or Click to Upload Revenue Ledger XLSX", className="upload-content")
            ]),
            multiple=False,
            className="upload-box"
        ),
        html.Div(id="upload_revenues_output", className="upload-output"),

        # Upload box for the enrollees ledger
        dcc.Upload(
            id="upload_enrollees",
            children=html.Div([
                html.Img(src="/assets/icons/xlsx_icon.png", className="upload-icon"),
                html.Div("Drag & Drop or Click to Upload Enrollee Information XLSX", className="upload-content")
            ]),
            multiple=False,
            className="upload-box"
        ),
        html.Div(id="upload_enrollees_output", className="upload-output"),

        html.Button("Confirm Uploads", id="upload_confirm_btn", className="setup-btn", disabled=True)
    ])
)

##############################################
# STEP 1 CALLBACKS - SAVE CONTENTS INTO STORES
##############################################
@dash_app.callback(
    [
        Output("upload_revenues", "style"),
        Output("upload_revenues_output", "children"),
        Output("stored_revenue", "data"),   # store contents
    ],
    Input("upload_revenues", "contents"),
    State("upload_revenues", "filename"),
    prevent_initial_call=True
)
def show_revenue_filename(contents, filename):
    if contents and filename:
        return (
            {"display": "none"},
            html.Div(
                className="file-display",
                children=[
                    html.Img(src="/assets/icons/xlsx_icon.png", className="file-icon"),
                    html.Span(filename, className="file-name"),
                    html.Button("X", id="delete_revenues", className="delete-btn")
                ]
            ),
            contents   # store the base64 string
        )
    return {"display": "block"}, None, None


@dash_app.callback(
    [
        Output("upload_enrollees", "style"),
        Output("upload_enrollees_output", "children"),
        Output("stored_enrollees", "data"),   # store contents
    ],
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
                    html.Img(src="/assets/icons/xlsx_icon.png", className="file-icon"),
                    html.Span(filename, className="file-name"),
                    html.Button("X", id="delete_enrollees", className="delete-btn")
                ]
            ),
            contents   # store the base64 string
        )
    return {"display": "block"}, None, None


##################################################
# STEP 1 CALLBACKS - Clear Files when X is clicked
##################################################

# Clear Revenue file when X is clicked
@dash_app.callback(
    [
        Output("upload_revenues", "style", allow_duplicate=True),
        Output("upload_revenues_output", "children", allow_duplicate=True),
        Output("upload_revenues", "contents", allow_duplicate=True),
    ],
    Input("delete_revenues", "n_clicks"),
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

######################################################################
# STEP 1 CALLBACKS - Enable "Next" button once both files are uploaded
######################################################################
@dash_app.callback(
    Output("upload_confirm_btn", "disabled"),
    [Input("upload_revenues", "contents"),
     Input("upload_enrollees", "contents")],
    prevent_initial_call=False
)
def enable_next_button(revenues_contents, enrollees_contents):
    # If both uploads have contents, enable the button
    if revenues_contents and enrollees_contents:
        return False   # not disabled
    return True        # keep disabled