import pandas as pd

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
from MachineLearning.Utils.run_models_parallel import SurvivalExperimentRunner

html_step_3 = (
    html.Div([
        html.Div("3", className="step-number"),
        html.H3("Waiting for Model Results", className="step-header"),
        html.Div("Training in progress..."),
        html.Button("Next", id="next_btn")
    ])
)

html_step_4 = (
    html.Div([
        html.Div("4", className="step-number"),
        html.H3("Model Result Analysis", className="step-header"),
        dcc.Dropdown(id="model_summary_dropdown", placeholder="Select a model"),
        dcc.Graph(id="auc_graph"),
        html.Button("Next", id="next_btn")
    ])
)

html_step_5 = (
    html.Div([
        html.Div("5", className="step-number"),
        html.H3("Finalization", className="step-header"),
        html.Button("Finalize Model", id="finalize_btn")
    ])
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

    cs = CreditSales(df_revenues, df_enrollees, args)
    df_credit_sales = cs.show_data()

    return df_credit_sales

def run_model_training(models_data, balancing_data):
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


def run_model_training(models_data, balancing_data):
    print("Running model training execution...")
    print("Models selected:", models_data)
    print("Balancing strategies:", balancing_data)
    # Training logic here


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