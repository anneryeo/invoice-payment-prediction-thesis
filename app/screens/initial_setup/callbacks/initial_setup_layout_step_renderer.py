from datetime import datetime

from dash import Input, Output, State, html, dcc, no_update
from app import dash_app

from app.screens.initial_setup.callbacks.step_1 import html_step_1
from app.screens.initial_setup.callbacks.step_2 import html_step_2
from app.screens.initial_setup.callbacks.step_3 import html_step_3, run_training
from app.screens.initial_setup.callbacks.step_4 import html_step_4
from app.screens.initial_setup.callbacks.step_5 import html_step_5, run_finalization


# ── Persistent root layout ────────────────────────────────────────────────────
# Mounted permanently in app.py outside the router so these store IDs and
# step-content always exist in the DOM regardless of which page is active.
# Do NOT redeclare these stores in InitialSetupScreen.layout() — they live here.
initial_setup_layout = html.Div([
    dcc.Store(id="current_step",        data="progress-1"),  # active step tracker
    dcc.Store(id="training_status"),                         # tracker for the machine learning process
    dcc.Store(id="stored_revenue"),                          # uploaded revenue file (base64)
    dcc.Store(id="stored_enrollees"),                        # uploaded enrollees file (base64)
    dcc.Store(id="stored_models"),                           # selected model types
    dcc.Store(id="stored_balancing"),                        # selected balancing strategies
    dcc.Store(id="stored_credit_sales"),                     # cleaned df_credit_sales from step 3
    dcc.Store(id="selected-model-data"),                     # final selected model for deployment
    dcc.Store(id="fin-training_status"),                     # tracker for the deployment steps
    html.Div(id="step-content"),                             # step layout rendered here
])


##############################################
# MAIN RENDER CALLBACK
##############################################

@dash_app.callback(
    Output("step-content", "children"),
    Input("current_step", "data"),
    prevent_initial_call=False,
)
def render_step(step):
    if step == "progress-1":
        return html_step_1
    elif step == "progress-2":
        return html_step_2
    elif step == "progress-3":
        return html_step_3
    elif step == "progress-4":
        return html_step_4
    elif step == "progress-5":
        return html_step_5


##############################################
# PROGRESS HEADER CLASS UPDATE
##############################################

@dash_app.callback(
    Output("progress-1", "className"),
    Output("progress-2", "className"),
    Output("progress-3", "className"),
    Output("progress-4", "className"),
    Output("progress-5", "className"),
    Input("current_step", "data"),
)
def update_progress_classes(current_step):
    step_num = int(current_step.split("-")[1])
    classes = []
    for i in range(1, 6):
        if i < step_num:
            classes.append("progress-step complete")
        elif i == step_num:
            classes.append("progress-step active")
        else:
            classes.append("progress-step future")
    return classes


##############################################
# STEP ADVANCEMENT CALLBACKS
##############################################

# --- Step 1 → Step 2 ---
@dash_app.callback(
    Output("current_step", "data", allow_duplicate=True),
    Input("upload_confirm_btn", "n_clicks"),
    prevent_initial_call=True,
)
def go_to_step_2(upload_clicks):
    if upload_clicks:
        return "progress-2"
    return no_update


# --- Step 2 → Step 3 (UI advance only, training fires separately) ---
# The code for the training is found in step_3.py
@dash_app.callback(
    Output("current_step", "data", allow_duplicate=True),
    Input("model_parameters_confirm_btn", "n_clicks"),
    State("current_step", "data"),
    prevent_initial_call=True,
)
def go_to_step_3(confirm_clicks, current_step):
    if confirm_clicks and current_step != "progress-3":
        return "progress-3"
    return no_update


# --- Step 3 → Step 4 ---
@dash_app.callback(
    Output("current_step", "data", allow_duplicate=True),
    Input("next_btn", "n_clicks"),
    prevent_initial_call=True,
)
def go_to_step_4(next_clicks):
    if next_clicks:
        return "progress-4"
    return no_update


# --- Step 4 → Step 5 ---
@dash_app.callback(
    Output("current_step", "data", allow_duplicate=True),
    Input("finalize_btn", "n_clicks"),
    prevent_initial_call=True,
)
def go_to_step_5(finalize_clicks):
    if finalize_clicks:
        return "progress-5"
    return no_update