from datetime import datetime

from dash import Input, Output, State, html, dcc, no_update
from app import dash_app

from app.screens.initial_setup.callbacks.step_1 import html_step_1
from app.screens.initial_setup.callbacks.step_2 import html_step_2
from app.screens.initial_setup.callbacks.step_3 import html_step_3, clean_datasets, run_model_training
from app.screens.initial_setup.callbacks.step_4 import html_step_4
from app.screens.initial_setup.callbacks.step_5 import html_step_5

from machine_learning.utils.features.adjust_survival_time_periods import adjust_payment_period
from machine_learning.utils.features.get_slope_time_points import get_slope_timepoints
from machine_learning.utils.training.tune_cox_hyperparameters import CoxHyperparameterTuner
from machine_learning.utils.training.run_models_parallel import progress_state
from machine_learning.utils.io.save_results_to_folder import save_training_results


# ── Persistent root layout ────────────────────────────────────────────────────
# Mounted permanently in app.py outside the router so these store IDs and
# step-content always exist in the DOM regardless of which page is active.
# Do NOT redeclare these stores in InitialSetupScreen.layout() — they live here.
initial_setup_layout = html.Div([
    dcc.Store(id="current_step",     data="progress-1"),  # active step tracker
    dcc.Store(id="training_status"),                      # ML pipeline status
    dcc.Store(id="stored_revenue"),                       # uploaded revenue file (base64)
    dcc.Store(id="stored_enrollees"),                     # uploaded enrollees file (base64)
    dcc.Store(id="stored_models"),                        # selected model types
    dcc.Store(id="stored_balancing"),                     # selected balancing strategies
    html.Div(id="step-content"),                          # step layout rendered here
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


@dash_app.callback(
    Output("training_status", "data"),
    Input("model_parameters_confirm_btn", "n_clicks"),
    State("stored_revenue", "data"),
    State("stored_enrollees", "data"),
    State("stored_models", "data"),
    State("stored_balancing", "data"),
    prevent_initial_call=True,
)
def run_training(confirm_clicks, revenue_data, enrollees_data, models_data, balancing_data):
    if not confirm_clicks:
        return no_update

    progress_state["extraction_done"] = False
    progress_state["survival_done"]   = False
    progress_state["training_done"]   = False
    progress_state["saving_done"]     = False
    start_time = datetime.now()

    print("Running training...")
    df_data, df_data_surv = clean_datasets(revenue_data, enrollees_data)

    print("Getting best parameters for the CoxPH Model...")
    X_surv = df_data_surv.drop(columns=["days_elapsed_until_fully_paid", "censor"])
    T      = adjust_payment_period(df_data_surv["days_elapsed_until_fully_paid"])
    E      = df_data_surv["censor"]

    tuner = CoxHyperparameterTuner(save_report_path="Results/")
    tuner.fit(X_surv, T, E)
    progress_state["survival_done"] = True

    survival_results_dict = {
        "best_c_index":          tuner.best_c_index_,
        "best_surv_parameters":  tuner.best_params_,
    }

    print("Proceeding to model training...")

    class Config:
        parameters_dir = r"machine_learning\parameters.json"
        target_feature = "dtp_bracket"
        test_size      = 0.3
        time_points    = get_slope_timepoints(T, E, n_points=9)

    args = Config()
    print(f"Using timepoints: {args.time_points}")

    model_results_df, class_mappings_dict = run_model_training(
        df_data, df_data_surv, models_data, balancing_data, args, tuner.best_params_
    )
    progress_state["training_done"] = True

    end_time            = datetime.now()
    total_training_time = end_time - start_time

    save_training_results(
        model_results_df,
        survival_results_dict,
        class_mappings_dict,
        "Results",
        models_data,
        start_time.isoformat(),
        end_time.isoformat(),
        str(total_training_time),
        format="sqlite",
    )
    progress_state["saving_done"] = True

    return "done"


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