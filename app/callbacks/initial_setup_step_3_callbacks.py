import base64
import io
import pandas as pd
from datetime import datetime
from time import time

import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, dcc, no_update

from app import dash_app
from feature_engineering.credit_sales_machine_learning import CreditSales
from machine_learning import (
    AdaBoostPipeline,
    DecisionTreePipeline,
    GaussianNaiveBayesPipeline,
    KnearestNeighborPipeline,
    RandomForestPipeline,
    XGBoostPipeline,
    MultiLayerPerceptronPipeline,
    TransformerPipeline,
)
from machine_learning.utils.training.run_models_parallel import SurvivalExperimentRunner, progress_state


# ══════════════════════════════════════════════════════════════════════════════
#  HTML LAYOUT — STEP 3
# ══════════════════════════════════════════════════════════════════════════════

html_step_3 = html.Div(
    className="sub-progress-tracker",
    children=[

        # ── TOP SECTION: 3-column vertical stepper ──────────────────────────
        html.Div(
            className="sub-stepper-container",
            children=[

                # Left column
                html.Div(className="sub-col-left", children=[
                    html.Div(className="sub-col-left-row sub-col-circle", children=[
                        html.Span("Extraction of Invoice Details.", id="step1-status", className="sub-label"),
                    ]),
                    html.Div(className="sub-col-left-row sub-col-connector"),
                    html.Div(className="sub-col-left-row sub-col-circle"),
                    html.Div(className="sub-col-left-row sub-col-connector"),
                    html.Div(className="sub-col-left-row sub-col-circle", children=[
                        html.Span("Training of Model/s.", id="step3-status", className="sub-label"),
                    ]),
                    html.Div(className="sub-col-left-row sub-col-connector"),
                    html.Div(className="sub-col-left-row sub-col-circle"),
                ]),

                # Center column
                html.Div(className="sub-col-center", children=[
                    html.Div(className="sub-circle future", id="sub-step1"),
                    html.Div(className="sub-connector"),
                    html.Div(className="sub-circle future", id="sub-step2"),
                    html.Div(className="sub-connector"),
                    html.Div(className="sub-circle future", id="sub-step3"),
                    html.Div(className="sub-connector"),
                    html.Div(className="sub-circle future", id="sub-step4"),
                ]),

                # Right column
                html.Div(className="sub-col-right", children=[
                    html.Div(className="sub-col-right-row sub-col-circle"),
                    html.Div(className="sub-col-right-row sub-col-connector"),
                    html.Div(className="sub-col-right-row sub-col-circle", children=[
                        html.Span("Train Survival Analysis Model.", id="step2-status", className="sub-label"),
                    ]),
                    html.Div(className="sub-col-right-row sub-col-connector"),
                    html.Div(className="sub-col-right-row sub-col-circle"),
                    html.Div(className="sub-col-right-row sub-col-connector"),
                    html.Div(className="sub-col-right-row sub-col-circle", children=[
                        html.Span("Save Model Results.", id="step4-status", className="sub-label"),
                    ]),
                ]),
            ],
        ),

        # ── BOTTOM SECTION: progress bar, text, button ──────────────────────
        html.Div(className="bottom-progress-container", children=[
            dbc.Progress(
                id="training-progress",
                value=0,
                striped=True,
                animated=True,
                color="primary",
                style={"height": "1.5rem"},
            ),
            html.Div(id="progress-text", className="status-message"),
            dcc.Interval(id="progress-interval", interval=1000, n_intervals=0),
            html.Button(
                "Compare Model Results",
                id="next_btn",
                className="compare-button",
                disabled=True,
            ),
        ]),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def clean_datasets(revenues_content, enrollees_content):
    print("Cleaning datasets and preparing them...")

    revenue_bytes = base64.b64decode(revenues_content.split(",")[1])
    enrollees_bytes = base64.b64decode(enrollees_content.split(",")[1])

    df_revenues = pd.read_excel(io.BytesIO(revenue_bytes))
    df_enrollees = pd.read_excel(io.BytesIO(enrollees_bytes))

    class Config:
        observation_end = datetime(2026, 3, 3, 23, 59, 59)
    args = Config()

    cs = CreditSales(df_revenues, df_enrollees, args,
                     drop_demographic_columns=True,
                     drop_helper_columns=True,
                     drop_plan_type_columns=True,
                     drop_missing_dtp=True)
    df_credit_sales = cs.show_data()

    progress_state["extraction_done"] = True

    survival_columns = ['days_elapsed_until_fully_paid', 'censor']
    non_survival_columns = ['due_date', 'dtp_bracket']

    df_data = df_credit_sales[df_credit_sales['censor'] == 1]
    df_data.drop(columns=survival_columns, inplace=True)
    df_data_surv = df_credit_sales.drop(columns=non_survival_columns)

    return df_data, df_data_surv


def run_model_training(df_data, df_data_surv, models_data, balancing_data, args, best_penalty):
    PIPELINE_MAP = {
        "ada_boost": AdaBoostPipeline,
        "decision_tree": DecisionTreePipeline,
        "gaussian_naive_bayes": GaussianNaiveBayesPipeline,
        "knn": KnearestNeighborPipeline,
        "random_forest": RandomForestPipeline,
        "xgboost": XGBoostPipeline,
        "nn_mlp": MultiLayerPerceptronPipeline,
        "nn_transformer": TransformerPipeline,
    }

    selected_pipelines = {m: PIPELINE_MAP[m] for m in models_data if m in PIPELINE_MAP}

    do_not_parallel_compute = ['nn_transformer']

    runner = SurvivalExperimentRunner(
        df_data=df_data,
        df_data_surv=df_data_surv,
        models=selected_pipelines,
        balance_strategies=balancing_data,
        args=args,
        best_penalty=best_penalty,
        thresholds=None,
        n_jobs=-1,
        do_not_parallel_compute=do_not_parallel_compute,
        feature_selection_baseline=True,
        feature_selection_enhanced=True,
    )

    json_results = runner.run()

    return json_results


def evaluate_model(models_data):
    print("Evaluating the trained model...")
    print("Models used:", models_data)


def generate_reports():
    print("Generating reports...")


def finalize_pipeline():
    print("Finalizing pipeline...")


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS — STEP 3
# ══════════════════════════════════════════════════════════════════════════════

progress_state["start_time"] = time()


# ── Circle colors ─────────────────────────────────────────────────────────────
@dash_app.callback(
    [Output("sub-step1", "className"),
     Output("sub-step2", "className"),
     Output("sub-step3", "className"),
     Output("sub-step4", "className")],
    Input("progress-interval", "n_intervals"),
)
def update_sub_steps(n):
    if progress_state.get("extraction_done"):
        step1_class = "sub-circle complete"
    elif progress_state.get("start_time"):
        step1_class = "sub-circle active"
    else:
        step1_class = "sub-circle future"

    if progress_state.get("survival_done"):
        step2_class = "sub-circle complete"
    elif progress_state.get("extraction_done"):
        step2_class = "sub-circle active"
    else:
        step2_class = "sub-circle future"

    completed = int(progress_state.get("completed", 0) or 0)
    total = int(progress_state.get("total", 0) or 0)
    if total > 0 and completed >= total:
        step3_class = "sub-circle complete"
    elif progress_state.get("survival_done"):
        step3_class = "sub-circle active"
    else:
        step3_class = "sub-circle future"

    if progress_state.get("saving_done"):
        step4_class = "sub-circle complete"
    elif total > 0 and completed >= total:
        step4_class = "sub-circle active"
    else:
        step4_class = "sub-circle future"

    return step1_class, step2_class, step3_class, step4_class


# ── Step status text ──────────────────────────────────────────────────────────
@dash_app.callback(
    [Output("step1-status", "children"),
     Output("step2-status", "children"),
     Output("step3-status", "children"),
     Output("step4-status", "children")],
    [Input("progress-interval", "n_intervals"),
     Input("stored_revenue", "data"),
     Input("stored_enrollees", "data")],
    prevent_initial_call=False,
)
def update_step_statuses(n, revenue_data, enrollees_data):
    # What it will render when the respective step is not in progress or not completed
    step1 = "Extraction of Invoice Details."
    step2 = "Train Survival Analysis Model."
    step3 = "Train All Models."
    step4 = "Save Model Results."

    try:
        # Step 1
        if revenue_data and enrollees_data:
            step1 = "Extracting invoice details..."
        if progress_state.get("extraction_done"):
            step1 = "Invoice details extracted."

        # Step 2
        if progress_state.get("survival_done"):
            step2 = "Survival analysis completed."
        elif progress_state.get("extraction_done"):
            step2 = "Training survival analysis model..."

        # Step 3
        completed = int(progress_state.get("completed", 0) or 0)
        total = int(progress_state.get("total", 0) or 0)
        if progress_state.get("training_done"):
            step3 = "Model training completed."
        elif progress_state.get("survival_done"):
            step3 = "Training various model/s..."

        # Step 4
        saving_done = bool(progress_state.get("saving_done", False))
        if saving_done:
            step4 = "Model results saved."
        elif total > 0 and completed >= total:
            step4 = "Saving model results..."

    except Exception as e:
        print(f"[update_step_statuses] Error: {e}")

    return step1, step2, step3, step4

# ── Progress bar + text + button enable ───────────────────────────────────────
@dash_app.callback(
    [Output("training-progress", "value"),
     Output("training-progress", "animated"),
     Output("training-progress", "color"),
     Output("progress-text", "children"),
     Output("next_btn", "disabled")],
    Input("progress-interval", "n_intervals"),
)
def update_progress(n):
    completed = int(progress_state.get("completed", 0) or 0)
    total = int(progress_state.get("total", 0) or 0)
    start_time = progress_state.get("start_time")

    if total > 0 and completed > 0 and start_time:
        elapsed = time() - start_time
        avg_per_exp = elapsed / completed
        remaining = avg_per_exp * (total - completed)

        if remaining < 60:
            eta_str = f"~{int(remaining)}s remaining"
        elif remaining < 3600:
            eta_str = f"~{int(remaining / 60)}m remaining"
        else:
            eta_str = f"~{int(remaining / 3600)}h remaining"

        percent = int((completed / total) * 100)
        if completed >= total:
            saving_done = bool(progress_state.get("saving_done", False))
            return 100, False, "success", f"All {total} models trained (100%)", not saving_done
        else:
            return percent, True, "primary", f"{completed}/{total} models trained ({percent}%) — {eta_str}", True

    return 0, True, "primary", "Waiting for experiments to start...", True