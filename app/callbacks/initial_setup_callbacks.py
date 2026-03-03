import pandas as pd

from dash import Input, Output, State, html, dcc, ctx, no_update
from app import dash_app

from app.callbacks.initial_setup_step_1_callbacks import (
    html_step_1,
    show_revenue_filename, show_enrollees_filename, clear_enrollees_file, enable_next_button
)

from app.callbacks.initial_setup_step_2_callbacks import (
    html_step_2,
    toggle_confirm_button
)

from app.callbacks.initial_setup_step_3_callbacks import (
    html_step_3,
    clean_datasets, run_model_training
)

from MachineLearning.Utils.calculate_best_penalty import calculate_best_penalty

# Layout screen layoot with hidden stores
initial_setup_layout = html.Div([
    dcc.Store(id="current_step", data="progress-1"),                # track active step
    dcc.Store(id="training_status"),                                # track the machine learning status
    dcc.Store(id="stored_revenue"),                                 # store revenue file (base64 format) 
    dcc.Store(id="stored_enrollees"),                               # store enrollees file (base64 format)
    dcc.Store(id="stored_models"),                                  # store selected models
    dcc.Store(id="stored_balancing"),                               # store selected balancing strategies
    html.Div(id="step-content")                                     # where steps render
])


##############################################
# MAIN RENDER CALLBACK
##############################################
@dash_app.callback(
    Output("step-content", "children"),
    Input("current_step", "data"),
    prevent_initial_call=False
)
def render_step(step):
    if step == "progress-1":
        return html_step_1
         
    elif step == "progress-2":
        return html_step_2
    
    elif step == "progress-3":
        return html_step_3

    elif step == "progress-4":
        return html.Div([
            html.Div("4", className="step-number"),
            html.H3("Model Result Analysis", className="step-header"),
            dcc.Dropdown(id="model_summary_dropdown", placeholder="Select a model"),
            dcc.Graph(id="auc_graph"),
            html.Button("Next", id="next_btn")
        ])

    elif step == "progress-5":
        return html.Div([
            html.Div("5", className="step-number"),
            html.H3("Finalization", className="step-header"),
            html.Button("Finalize Model", id="finalize_btn")
        ])


##############################################
# PROGRESS HEADER CLASS UPDATE
##############################################
@dash_app.callback(
    [Output("progress-1", "className"),
     Output("progress-2", "className"),
     Output("progress-3", "className"),
     Output("progress-4", "className"),
     Output("progress-5", "className")],
     Input("current_step", "data")
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
# STEP ADVANCEMENT CALLBACK WITH ACTIONS
##############################################

# Step 1 → Step 2
@dash_app.callback(
    Output("current_step", "data", allow_duplicate=True),
    Input("upload_confirm_btn", "n_clicks"),
    State("current_step", "data"),
    prevent_initial_call=True
)
def go_to_step_2(upload_clicks, current_step):
    if upload_clicks:
        return "progress-2"
    return current_step


# Step 2 → Step 3
# Step change only
# so that the UI loads imediately before precessing
@dash_app.callback(
    Output("current_step", "data", allow_duplicate=True),
    Input("model_parameters_confirm_btn", "n_clicks"),
    State("current_step", "data"),
    prevent_initial_call=True
)
def go_to_step_3(confirm_clicks, current_step):
    # Only advance if we are not already at step 3
    if confirm_clicks and current_step != "progress-3":
        return "progress-3"
    return current_step


@dash_app.callback(
    Output("training_status", "data"),
    Input("model_parameters_confirm_btn", "n_clicks"),
    State("stored_revenue", "data"),
    State("stored_enrollees", "data"),
    State("stored_models", "data"),
    State("stored_balancing", "data"),
    prevent_initial_call=True
)
def run_training(confirm_clicks, revenue_data, enrollees_data, models_data, balancing_data):
    if confirm_clicks:
        print("Running training...")
        df_data, df_data_surv = clean_datasets(revenue_data, enrollees_data)

        print("Getting best penalty...")
        best_penalty = calculate_best_penalty(df_data_surv)

        class Config:
            # Path to the machine learning model parameters
            parameters_dir = r"MachineLearning\parameters.json"

            # Class to predict
            target_feature = 'dtp_bracket'
            # Test size in %
            test_size = 0.3

            # Time points used in generating survival features
            # It's not until 120 since the earliest pre-payment is 288 days
            time_points = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450]
        args = Config()

        print("Proceeeding to model training...")
        run_model_training(df_data, df_data_surv, models_data, balancing_data, args, best_penalty)
    
        return "done"
    return no_update


# Step 3 → Step 4
@dash_app.callback(
    Output("current_step", "data", allow_duplicate=True),
    Input("next_btn", "n_clicks"),
    State("current_step", "data"),
    prevent_initial_call=True
)
def go_to_step_4(next_clicks, current_step):
    if next_clicks:
        return "progress-4"
    return current_step


# Step 4 → Step 5
@dash_app.callback(
    Output("current_step", "data", allow_duplicate=True),
    Input("finalize_btn", "n_clicks"),
    State("current_step", "data"),
    prevent_initial_call=True
)
def go_to_step_5(finalize_clicks, current_step):
    if finalize_clicks:
        return "progress-5"
    return current_step