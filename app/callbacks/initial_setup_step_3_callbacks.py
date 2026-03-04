import time
import base64
import io
import pandas as pd
from datetime import datetime

import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, dcc, no_update

from app import dash_app
from FeatureEngineering.credit_sales_machine_learning import CreditSales
from MachineLearning import (
    AdaBoostPipeline,
    DecisionTreePipeline,
    GaussianNaiveBayesPipeline,
    KnearestNeighborPipeline,
    RandomForestPipeline,
    XGBoostPipeline,
    MultiLayerPerceptronPipeline,
    TransformerPipeline,
)
from MachineLearning.Utils.run_models_parallel import SurvivalExperimentRunner, progress_state

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

                # Left column  (labels that sit on the LEFT side)
                html.Div(className="sub-col-left", children=[
                    html.Div(className="sub-col-left-row sub-col-top", children=[
                        html.Span("Extraction of Invoice Details.", id="step1-status", className="sub-label"),
                    ]),
                    html.Div(className="sub-col-left-row sub-col-mid"),
                    html.Div(className="sub-col-left-row sub-col-bot", children=[
                        html.Span("Training of Model/s.", className="sub-label"),
                    ]),
                ]),

                # Center column  (circles + connecting line)
                html.Div(className="sub-col-center", children=[
                    html.Div(className="sub-circle future", id="sub-step1"),
                    html.Div(className="sub-connector"),
                    html.Div(className="sub-circle future", id="sub-step2"),
                    html.Div(className="sub-connector"),
                    html.Div(className="sub-circle future", id="sub-step3"),
                ]),

                # Right column  (labels that sit on the RIGHT side)
                html.Div(className="sub-col-right", children=[
                    html.Div(className="sub-col-right-row sub-col-top"),
                    html.Div(className="sub-col-right-row sub-col-mid", children=[
                        html.Span("Training Survival Analysis Model.", id="step2-status", className="sub-label"),
                    ]),
                    html.Div(className="sub-col-right-row sub-col-bot"),
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

    # ✅ Signal step 1 complete
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
        # "nn_rnn": RecurrentNeuralNetworkPipeline,  # add when ready
    }

    selected_pipelines = {m: PIPELINE_MAP[m] for m in models_data if m in PIPELINE_MAP}

    # GPU models — avoid parallel compute to prevent bugs
    do_not_parallel_compute = ['xg_boost', 'nn_mlp', 'nn_transformer']

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
        output_path="MachineLearning/Results/model_results.xlsx",
        feature_selection_baseline=True,
        feature_selection_enhanced=True,
    )

    df_results = runner.run()


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

progress_state["start_time"] = time.time()

# ── Step status text ──────────────────────────────────────────────────────────
# Single callback owns both outputs — avoids duplicate-output conflict.
# Inputs: interval for polling + stored data to detect file upload immediately.
@dash_app.callback(
    [Output("sub-step1", "className"),
     Output("sub-step2", "className"),
     Output("sub-step3", "className")],
    Input("progress-interval", "n_intervals"),
)
def update_sub_steps(n):
    # Step 1
    if progress_state.get("extraction_done"):
        step1_class = "sub-circle complete"
    elif progress_state.get("start_time"):
        step1_class = "sub-circle active"
    else:
        step1_class = "sub-circle future"

    # Step 2
    if progress_state.get("survival_done"):
        step2_class = "sub-circle complete"
    elif progress_state.get("extraction_done"):
        step2_class = "sub-circle active"
    else:
        step2_class = "sub-circle future"

    # Step 3
    completed = int(progress_state.get("completed", 0) or 0)
    total = int(progress_state.get("total", 0) or 0)
    if total > 0 and completed >= total:
        step3_class = "sub-circle complete"
    elif progress_state.get("survival_done"):
        step3_class = "sub-circle active"
    else:
        step3_class = "sub-circle future"

    return step1_class, step2_class, step3_class


@dash_app.callback(
    [Output("step1-status", "children"),
     Output("step2-status", "children")],
    [Input("progress-interval", "n_intervals"),
     Input("stored_revenue", "data"),
     Input("stored_enrollees", "data")],
    prevent_initial_call=False,
)
def update_step_statuses(n, revenue_data, enrollees_data):
    step1 = "Extraction of Invoice Details."
    step2 = "Training Survival Analysis Model."

    if revenue_data and enrollees_data:
        step1 = "Extracting invoice details..."

    if progress_state.get("extraction_done"):
        step1 = "Invoice details extracted."

    if progress_state.get("survival_done"):
        step2 = "Survival analysis completed."
    elif progress_state.get("extraction_done"):
        step2 = "Training survival analysis model..."

    return step1, step2


# ── Progress bar + text + button enable ───────────────────────────────────────
@dash_app.callback(
    [Output("training-progress", "value"),
     Output("progress-text", "children"),
     Output("next_btn", "disabled")],
    Input("progress-interval", "n_intervals"),
)
def update_progress(n):
    completed = int(progress_state.get("completed", 0) or 0)
    total = int(progress_state.get("total", 0) or 0)
    start_time = progress_state.get("start_time")

    if total > 0 and completed > 0 and start_time:
        elapsed = time.time() - start_time
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
            return 100, f"All {total} models trained (100%)", False
        else:
            return percent, f"{completed}/{total} models trained ({percent}%) — {eta_str}", True

    return 0, "Waiting for experiments to start...", True