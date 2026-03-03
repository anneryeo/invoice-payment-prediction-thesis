import pandas as pd

import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, dcc
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
from MachineLearning.Utils.run_models_parallel import SurvivalExperimentRunner

# Progress bar component
html_step_3 = html.Div(
    className="step-container",
    children=[
        html.H3("Step 3: Model Training & Results", className="step-header"),
        dbc.Progress(id="training-progress", value=0, striped=True, animated=True, style={"height": "30px"}),
        html.Div(id="progress-text", className="status-message"),
        dcc.Interval(id="progress-interval", interval=1000, n_intervals=0),  # update every second
        html.Button("Next ➡", id="next_btn", className="next-button")
    ]
)


###################
#  Helper functions
###################
import base64
import io
import pandas as pd
from datetime import datetime


def clean_datasets(revenues_content, enrollees_content):
    print("Cleaning datasets and preparing them...")

    # Decode base64 string into bytes
    revenue_bytes = base64.b64decode(revenues_content.split(",")[1])
    enrollees_bytes = base64.b64decode(enrollees_content.split(",")[1])

    # Read into pandas directly from memory
    df_revenues = pd.read_excel(io.BytesIO(revenue_bytes))
    df_enrollees = pd.read_excel(io.BytesIO(enrollees_bytes))

    class Config:
        # Hard-coded full datetime (year, month, day, hour, minute, second)
        observation_end = datetime(2026, 3, 3, 23, 59, 59)
    args = Config()

    cs = CreditSales(df_revenues, df_enrollees, args,
                     drop_demographic_columns=True,
                     drop_helper_columns=True,
                     drop_plan_type_columns=True,
                     drop_missing_dtp=True)
    df_credit_sales = cs.show_data()

    survival_columns = ['censor', 'days_elapsed_until_fully_paid']
    non_survival_columns = ['due_date', 'dtp_bracket']

    df_data = df_credit_sales.drop(columns=survival_columns)
    # Filter only invoices that are fully paid:
    df_data = df_credit_sales[df_credit_sales['censor'] == 1]

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
    
    
    # Build dictionary of selected pipelines
    selected_pipelines = {m: PIPELINE_MAP[m] for m in models_data if m in PIPELINE_MAP}

    # Since these models are trained in the GPU, it's best
    # to not to parallel compute to avoid bugs
    do_not_parallel_compute = ['xg_boost', 'nn_mlp', 'nn_transformer']

    balance_strategies = balancing_data

    # Create an experiment runner instance
    runner = SurvivalExperimentRunner(
        df_data=df_data,
        df_data_surv=df_data_surv,
        models=selected_pipelines,
        balance_strategies=balance_strategies,
        args=args,
        best_penalty=best_penalty,
        thresholds=None,
        n_jobs=-1,
        do_not_parallel_compute=do_not_parallel_compute,

        output_path="MachineLearning/Results/model_results.xlsx",
        feature_selection_baseline=True,
        feature_selection_enhanced=True
    )

    # Run experiments
    df_results = runner.run()

def evaluate_model(models_data):
    print("Evaluating the trained model...")
    print("Models used:", models_data)
    # Evaluation logic here


def generate_reports():
    print("Generating reports...")
    # Reporting logic here


def finalize_pipeline():
    print("Finalizing pipeline...")
    # Finalization logic here


#########################################################
#  STEP 3 MODELS - Running the techniques
#########################################################
@dash_app.callback(
    Output("stored_credit_sales", "data"),   # store df_credit_sales
    Input("stored_enrollees", "data"),       # depends on enrollees upload
    Input("stored_revenues", "data"),        # depends on revenues upload
    prevent_initial_call=True
)
def store_credit_sales(enrollees_content, revenues_content):
    if enrollees_content and revenues_content:
        df_credit_sales = clean_datasets(revenues_content, enrollees_content)
        # Serialize DataFrame to JSON for storage
        return df_credit_sales.to_json(date_format="iso", orient="split")
    return None


@dash_app.callback(
    [Output("training-progress", "value"),
     Output("progress-text", "children")],
    Input("progress-interval", "n_intervals")
)
def update_progress(n):
    completed = progress_state.get("completed", 0)
    total = progress_state.get("total", 1)
    percent = int((completed / total) * 100)
    return percent, f"{completed}/{total} experiments completed ({percent}%)"