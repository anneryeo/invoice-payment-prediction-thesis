import pickle
import os
import io
from time import time

from app import dash_app
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, dcc, no_update

from lifelines import CoxPHFitter
from utils.data_loaders.read_settings_json import read_settings_json
from machine_learning.utils.io.load_results_from_folder import SessionStore
from machine_learning.utils.features.adjust_survival_time_periods import adjust_payment_period
from machine_learning.utils.training.tune_cox_hyperparameters import CoxHyperparameterTuner

from machine_learning import (
    AdaBoostPipeline,
    DecisionTreePipeline,
    GaussianNaiveBayesPipeline,
    KNearestNeighborPipeline,
    RandomForestPipeline,
    XGBoostPipeline,
    StackedEnsemblePipeline,
    MultiLayerPerceptronPipeline,
    TransformerPipeline,
)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML LAYOUT — STEP 5
# ══════════════════════════════════════════════════════════════════════════════

html_step_5 = html.Div(
    className="sub-progress-tracker",
    children=[

        # ── TOP SECTION: 3-column vertical stepper ──────────────────────────
        html.Div(
            className="sub-stepper-container",
            children=[

                # Left column
                html.Div(className="sub-col-left", children=[
                    html.Div(className="sub-col-left-row sub-col-circle", children=[
                        html.Span("Train Selected Model.", id="fin-step1-status", className="sub-label"),
                    ]),
                    html.Div(className="sub-col-left-row sub-col-connector"),
                    html.Div(className="sub-col-left-row sub-col-circle"),
                    html.Div(className="sub-col-left-row sub-col-connector"),
                    html.Div(className="sub-col-left-row sub-col-circle", children=[
                        html.Span("Save Selected Model Weights.", id="fin-step3-status", className="sub-label"),
                    ]),
                    html.Div(className="sub-col-left-row sub-col-connector"),
                    html.Div(className="sub-col-left-row sub-col-circle"),
                ]),

                # Center column
                html.Div(className="sub-col-center", children=[
                    html.Div(className="sub-circle future", id="fin-sub-step1"),
                    html.Div(className="sub-connector"),
                    html.Div(className="sub-circle future", id="fin-sub-step2"),
                    html.Div(className="sub-connector"),
                    html.Div(className="sub-circle future", id="fin-sub-step3"),
                    html.Div(className="sub-connector"),
                    html.Div(className="sub-circle future", id="fin-sub-step4"),
                ]),

                # Right column
                html.Div(className="sub-col-right", children=[
                    html.Div(className="sub-col-right-row sub-col-circle"),
                    html.Div(className="sub-col-right-row sub-col-connector"),
                    html.Div(className="sub-col-right-row sub-col-circle", children=[
                        html.Span("Train Survival Analysis Model.", id="fin-step2-status", className="sub-label"),
                    ]),
                    html.Div(className="sub-col-right-row sub-col-connector"),
                    html.Div(className="sub-col-right-row sub-col-circle"),
                    html.Div(className="sub-col-right-row sub-col-connector"),
                    html.Div(className="sub-col-right-row sub-col-circle", children=[
                        html.Span("Save Survival Model Weights.", id="fin-step4-status", className="sub-label"),
                    ]),
                ]),
            ],
        ),

        # ── BOTTOM SECTION: progress bar, text, button ──────────────────────
        html.Div(className="bottom-progress-container", children=[
            dbc.Progress(
                id="fin-training-progress",
                value=0,
                striped=True,
                animated=True,
                color="primary",
                style={"height": "1.5rem"},
            ),
            html.Div(id="fin-progress-text", className="status-message"),
            dcc.Interval(id="fin-progress-interval", interval=1000, n_intervals=0),
            html.Button(
                "Finalize Model",
                id="finalize_btn",
                className="compare-button",
                disabled=True,
            ),
        ]),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED PROGRESS STATE — STEP 5
# ══════════════════════════════════════════════════════════════════════════════

fin_progress_state = {
    "start_time":     None,
    "selected_done":  False,
    "survival_done":  False,
    "selected_saved": False,
    "survival_saved": False,
}


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE MAP
# ══════════════════════════════════════════════════════════════════════════════

PIPELINE_MAP = {
    "ada_boost":            AdaBoostPipeline,
    "decision_tree":        DecisionTreePipeline,
    "gaussian_naive_bayes": GaussianNaiveBayesPipeline,
    "knn":                  KNearestNeighborPipeline,
    "random_forest":        RandomForestPipeline,
    "xgboost":              XGBoostPipeline,
    "stacked_ensemble":     StackedEnsemblePipeline,
    "nn_mlp":               MultiLayerPerceptronPipeline,
    "nn_transformer":       TransformerPipeline,
}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _train_selected_model(
    df_data, df_data_surv,
    model_key, balance_strategy,
    args, best_surv_params,
    fitted_cph=None,
):
    """
    Train only the single model chosen in Step 4 on the FULL dataset.

    No train/test split is performed here.  Performance evaluation already
    happened in Step 3 with a proper held-out set — those metrics are the
    ones the user saw in Step 4 when selecting this model.  The purpose of
    Step 5 is to produce the strongest possible model for deployment, so we
    use every available labelled example.

    Workflow
    --------
    1. Use all of df_data (no split).
    2. Encode labels, resample, and normalize via DataPreparer —
       X_test / y_test are left as None so normalize() skips test-set
       transformation entirely.
    3. Generate survival-enhanced features for the full prepared set,
       reusing the pre-fitted CoxPHFitter from Step 3.
    4. Instantiate the pipeline class directly and call fit() only —
       evaluate() is intentionally skipped (no held-out set exists).

    Parameters
    ----------
    df_data : pd.DataFrame
        Classification-ready dataset (survival columns already dropped).
    df_data_surv : pd.DataFrame
        Dataset with survival columns intact (used by generate_survival_features).
    model_key : str
        Key into PIPELINE_MAP identifying which algorithm to train.
    balance_strategy : str
        Resampling strategy name passed to DataPreparer.resample().
    args : Config
        Namespace with target_feature, time_points, parameters_dir.
    best_surv_params : dict
        CoxPH hyperparameters passed to generate_survival_features when
        fitted_cph is None (fresh-fit fallback only).
    fitted_cph : CoxPHFitter, optional
        Pre-fitted Cox model from Step 3. When supplied, fitting is skipped
        inside generate_survival_features — the model is reused as-is.

    Returns
    -------
    pipeline : BasePipeline subclass
        Fitted pipeline instance (.model and .features are populated;
        .results is None because evaluate() is not called).
    label_encoder : LabelEncoder
        Fitted encoder from DataPreparer — required at inference time to
        decode integer predictions back to original class labels.
    """
    from machine_learning.utils.features.generate_survival_features import generate_survival_features
    from machine_learning.utils.data.data_preparation import DataPreparer

    pipeline_cls = PIPELINE_MAP.get(model_key)
    if pipeline_cls is None:
        raise ValueError(f"Unknown model key: {model_key!r}")

    # ── 1. Encode, resample, normalize — no split ────────────────────────────
    # X_test / y_test are left as None (their default in DataPreparer).
    # normalize() now guards against None, so no placeholder copies are needed.
    preparer = DataPreparer(
        df_data=df_data,
        target_feature=args.target_feature,
        verbose=True,
    )
    preparer.prep_full_data(balance_strategy=balance_strategy)

    X_bal = preparer.X_train
    y_bal = preparer.y_train

    # ── 2. Survival-enhanced features for the full balanced set ─────────────
    X_surv = df_data_surv.drop(columns=["days_elapsed_until_fully_paid", "censor"])
    T      = adjust_payment_period(df_data_surv["days_elapsed_until_fully_paid"])
    E      = df_data_surv["censor"]

    X_enhanced = generate_survival_features(
        X_surv=X_surv,
        T=T,
        E=E,
        X_train=X_bal,
        X_test=None,                    # no held-out set needed
        best_params=best_surv_params,   # ignored when fitted_cph is supplied
        time_points=args.time_points,
        fitted_cph=fitted_cph,          # reuse Step 3's Cox model
    )

    # ── 3. Instantiate pipeline and fit — no evaluate() ─────────────────────
    # BasePipeline requires X_test / y_test at construction; pass the training
    # data as a placeholder so the object is valid, but evaluate() is never
    # called — there is no honest held-out set at this stage.
    pipeline = pipeline_cls(
        X_train=X_enhanced,
        X_test=X_enhanced,
        y_train=y_bal,
        y_test=y_bal,
        args=args,
        parameters=getattr(args, "parameters", {}),
    )

    pipeline.initialize_model()
    pipeline.fit(use_feature_selection=True)
    # evaluate() intentionally omitted — metrics come from Step 3

    fin_progress_state["selected_done"] = True

    return pipeline, preparer.label_encoder


def _train_survival_model(df_data_surv, results_root: str) -> dict:
    """
    Re-fit the CoxPH model using the best hyperparameters that were already
    found during Step 3 (stored in SQLite).  This avoids a redundant
    hyperparameter search while still producing a serialisable fitted object
    that can be pickled.

    Step 3 only persists plain JSON (best_surv_parameters, best_time_points,
    best_c_index) — it never saves the fitted CoxPHFitter itself — so we must
    re-fit here.  Using the known-optimal parameters means this is a single
    deterministic fit, not a full grid/random search.

    Returns a dict with keys:
        cph                   – fitted CoxPHFitter
        best_c_index          – float
        best_surv_parameters  – dict  (penalizer, l1_ratio, ...)
        best_time_points      – list[int]
    """
    session          = SessionStore(results_root).load()
    survival_results = session["survival_results"]
    best_params      = survival_results["best_surv_parameters"]
    best_time_points = survival_results["best_time_points"]

    print(f"[Step 5] Re-fitting Cox model with Step 3 params: {best_params}")

    # Apply the same time adjustment that was used during Step 3 training,
    # then build a clean DataFrame for CoxPHFitter (features + T + E only).
    df_fit = df_data_surv.copy()
    df_fit["days_elapsed_until_fully_paid"] = adjust_payment_period(
        df_fit["days_elapsed_until_fully_paid"]
    )

    # CoxPHFitter splits params across __init__ and .fit()
    _INIT_PARAMS = {"penalizer", "l1_ratio", "baseline_estimation_method"}
    _FIT_PARAMS  = {"robust"}  # only robust goes directly
    _FIT_OPTIONS = {"step_size"}  # step_size must go inside fit_options

    init_kwargs  = {k: v for k, v in best_params.items() if k in _INIT_PARAMS}
    fit_kwargs   = {k: v for k, v in best_params.items() if k in _FIT_PARAMS}
    fit_options  = {k: v for k, v in best_params.items() if k in _FIT_OPTIONS}

    cph = CoxPHFitter(**init_kwargs)
    cph.fit(
        df_fit,
        duration_col="days_elapsed_until_fully_paid",
        event_col="censor",
        **fit_kwargs,
        fit_options=fit_options
    )

    fin_progress_state["survival_done"] = True

    return {
        "cph":                  cph,
        "best_c_index":         cph.concordance_index_,
        "best_surv_parameters": best_params,
        "best_time_points":     best_time_points,
    }

def _save_pickle(obj, path: str) -> None:
    """Persist an object to a pickle file."""
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS — STEP 5
# ══════════════════════════════════════════════════════════════════════════════

_finalization_triggered = False  # module-level guard

@dash_app.callback(
    Output("fin-training_status", "data"),
    Input("current_step",        "data"),       # ← fire when step changes
    State("selected-model-data", "data"),
    State("stored_credit_sales", "data"),
    prevent_initial_call=True,
)
def run_finalization(current_step, selected_model_data, credit_sales_json):
    global _finalization_triggered

    # Only act when we've just arrived at step 5
    if current_step != "progress-5":
        return no_update

    if _finalization_triggered:
        return no_update

    if not selected_model_data:
        print("[Step 5] No selected model data found — aborting.")
        return "error"

    if not credit_sales_json:
        print("[Step 5] stored_credit_sales is empty — was Step 3 completed?")
        return "error"

    _finalization_triggered = True
    for key in ("selected_done", "survival_done", "selected_saved", "survival_saved"):
        fin_progress_state[key] = False
    fin_progress_state["start_time"] = time()

    model_key        = selected_model_data.get("key")
    balance_strategy = selected_model_data.get("strategy")
    model_parameters = selected_model_data.get("parameters", {})

    print(f"[Step 5] Finalizing model: {model_key}  strategy: {balance_strategy}")

    try:
        import pandas as pd

        settings        = read_settings_json()
        results_root    = settings["Training"]["RESULTS_ROOT"]
        deployed_models = settings["Training"]["DEPLOYED_MODELS"]
        os.makedirs(deployed_models, exist_ok=True)

        # ── 1. Reconstruct datasets from stored df_credit_sales ──────────────
        df_credit_sales      = pd.read_json(io.StringIO(credit_sales_json), orient='split')
        survival_columns     = ['days_elapsed_until_fully_paid', 'censor']
        non_survival_columns = ['due_date', 'dtp_bracket']
        df_data      = df_credit_sales[df_credit_sales['censor'] == 1].copy()
        df_data.drop(columns=survival_columns, inplace=True)
        df_data_surv = df_credit_sales.drop(columns=non_survival_columns)

        # ── 2. Re-fit the survival model using Step 3's optimal parameters ───
        print("[Step 5] Re-fitting survival model with Step 3 hyperparameters...")
        survival_info    = _train_survival_model(df_data_surv, results_root)
        fitted_cph       = survival_info["cph"]
        best_surv_params = survival_info["best_surv_parameters"]
        best_time_points = survival_info["best_time_points"]

        # ── 3. Build args ─────────────────────────────────────────────────────
        class Config:
            parameters_dir = settings["Training"]["MODEL_PARAMETERS"]
            target_feature = settings["Training"]["target_feature"]
            time_points    = best_time_points
            parameters     = model_parameters
        args = Config()

        # ── 4. Train the selected classification model on the full dataset ────
        print(f"[Step 5] Training selected model ({model_key}) on full dataset...")
        pipeline, label_encoder = _train_selected_model(
            df_data, df_data_surv,
            model_key, balance_strategy,
            args, best_surv_params,
            fitted_cph=fitted_cph,
        )

        # ── 5. Save selected model weights ────────────────────────────────────
        selected_model_path = os.path.join(deployed_models, f"finalized_{model_key}.pkl")
        print(f"[Step 5] Saving selected model to {selected_model_path}...")
        _save_pickle(
            {
                "pipeline":      pipeline,
                "parameters":    model_parameters,
                "features":      pipeline.features,
                "label_encoder": label_encoder,
            },
            selected_model_path,
        )
        fin_progress_state["selected_saved"] = True

        # ── 6. Save survival model weights ────────────────────────────────────
        survival_model_path = os.path.join(deployed_models, "finalized_survival_model.pkl")
        print(f"[Step 5] Saving survival model to {survival_model_path}...")
        _save_pickle(
            {
                "tuner": survival_info["cph"],
                "survival_results": {
                    "best_c_index":         survival_info["best_c_index"],
                    "best_surv_parameters": best_surv_params,
                    "best_time_points":     best_time_points,
                },
            },
            survival_model_path,
        )
        fin_progress_state["survival_saved"] = True

        print("[Step 5] Finalization complete.")
        return "done"

    except Exception:
        import traceback
        print(f"[Step 5] FATAL ERROR in run_finalization:\n{traceback.format_exc()}")
        return "error"



# ── Circle colors ─────────────────────────────────────────────────────────────
@dash_app.callback(
    [Output("fin-sub-step1", "className"),
     Output("fin-sub-step2", "className"),
     Output("fin-sub-step3", "className"),
     Output("fin-sub-step4", "className")],
    Input("fin-progress-interval", "n_intervals"),
    prevent_initial_call=True,
)
def update_fin_sub_steps(n):
    if fin_progress_state.get("selected_done"):
        step1_class = "sub-circle complete"
    elif fin_progress_state.get("start_time"):
        step1_class = "sub-circle active"
    else:
        step1_class = "sub-circle future"

    if fin_progress_state.get("survival_done"):
        step2_class = "sub-circle complete"
    elif fin_progress_state.get("selected_done"):
        step2_class = "sub-circle active"
    else:
        step2_class = "sub-circle future"

    if fin_progress_state.get("selected_saved"):
        step3_class = "sub-circle complete"
    elif fin_progress_state.get("survival_done"):
        step3_class = "sub-circle active"
    else:
        step3_class = "sub-circle future"

    if fin_progress_state.get("survival_saved"):
        step4_class = "sub-circle complete"
    elif fin_progress_state.get("selected_saved"):
        step4_class = "sub-circle active"
    else:
        step4_class = "sub-circle future"

    return step1_class, step2_class, step3_class, step4_class


# ── Step status text ──────────────────────────────────────────────────────────
@dash_app.callback(
    [Output("fin-step1-status", "children"),
     Output("fin-step2-status", "children"),
     Output("fin-step3-status", "children"),
     Output("fin-step4-status", "children")],
    Input("fin-progress-interval", "n_intervals"),
    prevent_initial_call=True,
)
def update_fin_step_statuses(n):
    step1 = "Train Selected Model."
    step2 = "Train Survival Analysis Model."
    step3 = "Save Selected Model Weights."
    step4 = "Save Survival Model Weights."

    try:
        if fin_progress_state.get("start_time") and not fin_progress_state.get("selected_done"):
            step1 = "Training selected model..."
        elif fin_progress_state.get("selected_done"):
            step1 = "Selected model trained."

        if fin_progress_state.get("survival_done"):
            step2 = "Survival model trained."
        elif fin_progress_state.get("selected_done"):
            step2 = "Re-fitting survival model..."

        if fin_progress_state.get("selected_saved"):
            step3 = "Selected model weights saved."
        elif fin_progress_state.get("survival_done"):
            step3 = "Saving selected model weights..."

        if fin_progress_state.get("survival_saved"):
            step4 = "Survival model weights saved."
        elif fin_progress_state.get("selected_saved"):
            step4 = "Saving survival model weights..."

    except Exception as e:
        print(f"[update_fin_step_statuses] Error: {e}")

    return step1, step2, step3, step4


# ── Progress bar + text + button enable ───────────────────────────────────────
@dash_app.callback(
    [Output("fin-training-progress", "value"),
     Output("fin-training-progress", "animated"),
     Output("fin-training-progress", "color"),
     Output("fin-progress-text",     "children"),
     Output("finalize_btn",          "disabled")],
    Input("fin-progress-interval", "n_intervals"),
    prevent_initial_call=True,
)
def update_fin_progress(n):
    start_time = fin_progress_state.get("start_time")

    milestones = [
        fin_progress_state.get("selected_done"),
        fin_progress_state.get("survival_done"),
        fin_progress_state.get("selected_saved"),
        fin_progress_state.get("survival_saved"),
    ]
    completed = sum(bool(m) for m in milestones)
    total     = len(milestones)
    percent   = int((completed / total) * 100)

    if completed >= total:
        return 100, False, "success", "Finalization complete. Model is ready.", False

    if completed > 0 and start_time:
        elapsed      = time() - start_time
        avg_per_step = elapsed / completed
        remaining    = avg_per_step * (total - completed)

        if remaining < 60:
            eta_str = f"~{int(remaining)}s remaining"
        elif remaining < 3600:
            eta_str = f"~{int(remaining / 60)}m remaining"
        else:
            eta_str = f"~{int(remaining / 3600)}h remaining"

        return percent, True, "primary", f"{completed}/{total} steps completed ({percent}%) — {eta_str}", True

    return 0, True, "primary", "Waiting for finalization to start...", True