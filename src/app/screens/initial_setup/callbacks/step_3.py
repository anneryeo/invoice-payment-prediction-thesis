import base64
import io
import logging
import os
import pickle
import pandas as pd
from datetime import datetime
from time import time

from src.app import dash_app
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, no_update

from src.utils.data_loaders.read_settings_json import read_settings_json

from src.modules.feature_engineering.credit_sales_machine_learning import CreditSalesProcessor
from src.modules.machine_learning import (
    AdaBoostPipeline,
    DecisionTreePipeline,
    GaussianNaiveBayesPipeline,
    KNearestNeighborPipeline,
    RandomForestPipeline,
    XGBoostPipeline,
    StackedEnsemblePipeline,
    OrdinalPipeline,
    TwoStagePipeline,
    #MultiLayerPerceptronPipeline,
    #TransformerPipeline,
)
from src.modules.machine_learning.utils.features.adjust_survival_time_periods import adjust_payment_period
from src.modules.machine_learning.utils.features.get_slope_time_points import get_slope_timepoints
from src.modules.machine_learning.utils.training.tune_cox_hyperparameters import CoxHyperparameterTuner

from src.modules.machine_learning.utils.training.run_models_parallel import SurvivalExperimentRunner, progress_state
from src.modules.machine_learning.utils.io.save_results_to_folder import save_training_results

# Prevent loky (joblib's process backend) from spawning a wmic/powershell
# subprocess to count physical cores, which fails under Dash's restricted
# stdout environment. Must be set before any sklearn/sksurv/joblib import.
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())
logger = logging.getLogger(__name__)


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

    df_revenues  = pd.read_excel(io.BytesIO(revenue_bytes),  engine="calamine")
    df_enrollees = pd.read_excel(io.BytesIO(enrollees_bytes), engine="calamine")

    # Load observation_end from settings
    settings = read_settings_json()
    observation_end = settings["Training"]["observation_end"]
    # Convert string date to datetime
    parsed_date = datetime.strptime(observation_end, "%Y/%m/%d")

    class Config:
        observation_end = parsed_date
    args = Config()

    cs = CreditSalesProcessor(df_revenues, df_enrollees, args,
                    drop_helper_columns=True,
                    drop_demographic_columns=True,
                    drop_plan_type_columns=False,
                    drop_missing_dtp=True,
                    drop_back_account_transactions=True,
                    exclude_school_years=[2016, 2017, 2018],
                    winsorise_dtp=True)
    df_credit_sales = cs.show_data()

    progress_state["extraction_done"] = True

    survival_columns = ['days_elapsed_until_fully_paid', 'censor']
    non_survival_columns = ['due_date', 'dtp_bracket']

    df_data = df_credit_sales[df_credit_sales['censor'] == 1].copy()
    df_data.drop(columns=survival_columns, inplace=True)
    df_data_surv = df_credit_sales.drop(columns=non_survival_columns)

    return df_data, df_data_surv, df_credit_sales


def tune_cox_model(df_data_surv):
    X_surv = df_data_surv.drop(columns=["days_elapsed_until_fully_paid", "censor"])
    T      = adjust_payment_period(df_data_surv["days_elapsed_until_fully_paid"])
    E      = df_data_surv["censor"]

    # Focused grid based on empirical results — high l1_ratio wins consistently
    tuner = CoxHyperparameterTuner(
        alpha_grid       = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
        l1_ratios        = [0.5, 0.75, 1.0],
        save_report_path = "results/",
    )
    tuner.fit(X_surv, T, E)
    progress_state["survival_done"] = True

    best_time_points = get_slope_timepoints(T, E, n_points=9)

    # best_params_ now contains {"alpha": float, "l1_ratio": float}
    survival_results_dict = {
        "best_c_index":         tuner.best_c_index_,
        "best_surv_parameters": tuner.best_params_,
        "best_time_points":     best_time_points,
    }

    return survival_results_dict


def run_model_training(df_data, df_data_surv, models_data, balancing_data, args, best_parameters):
    PIPELINE_MAP = {
        "ada_boost": AdaBoostPipeline,
        "decision_tree": DecisionTreePipeline,
        "gaussian_naive_bayes": GaussianNaiveBayesPipeline,
        "knn": KNearestNeighborPipeline,
        "random_forest": RandomForestPipeline,
        "xgboost": XGBoostPipeline,
        "stacked_ensemble": StackedEnsemblePipeline,
        #"nn_mlp": MultiLayerPerceptronPipeline,
        #"nn_transformer": TransformerPipeline,
        "ordinal": OrdinalPipeline,
        "two_stage": TwoStagePipeline
    }

    selected_pipelines = {}
    missing_models = []

    for m in models_data:
        if m in PIPELINE_MAP:
            selected_pipelines[m] = PIPELINE_MAP[m]
        else:
            missing_models.append(m)

    if missing_models:
        print("Models not found:", ", ".join(missing_models))

    do_not_parallel_compute = ['nn_transformer']

    runner = SurvivalExperimentRunner(
        df_data=df_data,
        df_data_surv=df_data_surv,
        models=selected_pipelines,
        balance_strategies=balancing_data,
        args=args,
        best_parameters=best_parameters,
        thresholds=None,
        n_jobs=-1,
        do_not_parallel_compute=do_not_parallel_compute,
        feature_selection_baseline=True,
        feature_selection_enhanced=True,
        use_lda=True,
    )

    json_results = runner.run()

    return json_results


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS — STEP 3
# ══════════════════════════════════════════════════════════════════════════════

progress_state["start_time"] = time()


@dash_app.callback(
    Output("training_status", "data"),
    Output("stored_credit_sales", "data"),
    Input("current_step", "data"),
    State("stored_revenue", "data"),
    State("stored_enrollees", "data"),
    State("stored_models", "data"),
    State("stored_balancing", "data"),
    prevent_initial_call=True,
)
def run_training(current_step, revenue_data, enrollees_data, models_data, balancing_data):
    if current_step != "progress-3":
        return no_update, no_update

    progress_state["extraction_done"] = False
    progress_state["survival_done"]   = False
    progress_state["training_done"]   = False
    progress_state["saving_done"]     = False
    start_time = datetime.now()

    settings   = read_settings_json()
    debug_mode = settings["Config"].get("debug_mode", "False").strip().lower() == "true"
    temp_cache = settings["Config"].get("TEMP_CACHE", "temp_cache")
    cache_path = os.path.join(temp_cache, "step3_debug_cache.pkl")

    # ── PROFILER ─────────────────────────────────────────────────────────────
    _t_total = time()

    if debug_mode and os.path.exists(cache_path):
        logger.debug("Loading cached data from %s ...", cache_path)
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        df_credit_sales       = cache["df_credit_sales"]
        df_data               = cache["df_data"]
        df_data_surv          = cache["df_data_surv"]
        survival_results_dict = cache["survival_results_dict"]
        stored_credit_sales   = df_credit_sales.to_json(date_format='iso', orient='split')
        _cs_path = os.path.join(settings['Training']['RESULTS_ROOT'], 'credit_sales_cache.pkl')
        import pickle as _pkl
        with open(_cs_path, 'wb') as _fh:
            _pkl.dump(df_credit_sales, _fh)
        print(f'[step3] Saved credit_sales_cache.pkl to {_cs_path}')
        progress_state["extraction_done"] = True
        progress_state["survival_done"]   = True
        logger.debug("Cache loaded successfully; skipping extraction and Cox tuning.")
        print(f"[TIMING] cache_load: {time() - _t_total:.1f}s")
    else:
        print("Running training...")

        _t = time()
        df_data, df_data_surv, df_credit_sales = clean_datasets(revenue_data, enrollees_data)
        stored_credit_sales = df_credit_sales.to_json(date_format='iso', orient='split')
        _cs_path = os.path.join(settings['Training']['RESULTS_ROOT'], 'credit_sales_cache.pkl')
        import pickle as _pkl
        with open(_cs_path, 'wb') as _fh:
            _pkl.dump(df_credit_sales, _fh)
        print(f'[step3] Saved credit_sales_cache.pkl to {_cs_path}')
        print(f"[TIMING] clean_datasets: {time() - _t:.1f}s")

        _t = time()
        survival_results_dict = tune_cox_model(df_data_surv)
        print(f"[TIMING] tune_cox_model: {time() - _t:.1f}s")

        if debug_mode:
            os.makedirs(temp_cache, exist_ok=True)
            logger.debug("Saving data to cache at %s ...", cache_path)
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "df_credit_sales":       df_credit_sales,
                    "df_data":               df_data,
                    "df_data_surv":          df_data_surv,
                    "survival_results_dict": survival_results_dict,
                }, f)
            logger.debug("Cache saved successfully.")

    best_surv_params = survival_results_dict['best_surv_parameters']
    best_time_points = survival_results_dict['best_time_points']

    print("Proceeding to model training...")

    class Config:
        parameters_dir = settings["Training"]["MODEL_PARAMETERS"]
        target_feature = settings["Training"]["target_feature"]
        test_size      = float(settings["Training"]["test_size"])
        time_points    = best_time_points
    args = Config()
    print(f"Using timepoints: {best_time_points}")

    _t = time()
    model_results_df, class_mappings_dict = run_model_training(
        df_data, df_data_surv, models_data, balancing_data, args, best_surv_params
    )
    print(f"[TIMING] run_model_training: {time() - _t:.1f}s")
    progress_state["training_done"] = True

    end_time            = datetime.now()
    total_training_time = end_time - start_time

    _t = time()
    save_training_results(
        model_results_df,
        survival_results_dict,
        class_mappings_dict,
        settings['Training']['RESULTS_ROOT'],
        models_data,
        start_time.isoformat(),
        end_time.isoformat(),
        str(total_training_time),
        format="sqlite",
    )
    print(f"[TIMING] save_training_results: {time() - _t:.1f}s")
    progress_state["saving_done"] = True

    print(f"[TIMING] total run_training: {time() - _t_total:.1f}s")
    # ── END PROFILER ──────────────────────────────────────────────────────────

    return "done", stored_credit_sales


# ── Circle colors ─────────────────────────────────────────────────────────────
@dash_app.callback(
    [Output("sub-step1", "className"),
     Output("sub-step2", "className"),
     Output("sub-step3", "className"),
     Output("sub-step4", "className")],
    Input("progress-interval", "n_intervals"),
    State("current_step",      "data"),
    prevent_initial_call=True,
)
def update_sub_steps(n, step):
    if step != "progress-3":
        return no_update, no_update, no_update, no_update
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
    Input("progress-interval", "n_intervals"),
    State("current_step",      "data"),
    prevent_initial_call=True,
)
def update_step_statuses(n, step):
    if step != "progress-3":
        return no_update, no_update, no_update, no_update
    # What it will render when the respective step is not in progress or not completed
    step1 = "Extraction of Invoice Details."
    step2 = "Train Survival Analysis Model."
    step3 = "Train All Models."
    step4 = "Save Model Results."

    try:
        # Step 1
        if progress_state.get("start_time") and not progress_state.get("extraction_done"):
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
    State("current_step",      "data"),
    prevent_initial_call=True,
)
def update_progress(n, step):
    if step != "progress-3":
        return no_update, no_update, no_update, no_update, no_update
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

