import pickle
import threading
import traceback
import os
import io
from time import time

from app import dash_app
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, no_update

from utils.data_loaders.read_settings_json import read_settings_json
from machine_learning.utils.io.load_results_from_folder import SessionStore
from machine_learning.utils.features.adjust_survival_time_periods import adjust_payment_period
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
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
    OrdinalPipeline,
    TwoStagePipeline,
)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML LAYOUT — STEP 5
# ══════════════════════════════════════════════════════════════════════════════
#
# NOTE — fin-progress-interval is intentionally absent from this layout.
# It lives in the persistent root layout (initial_setup_layout_step_renderer.py)
# so that render_step can reliably enable it by returning disabled=False in the
# same response that mounts this layout.  Writing disabled=False to a
# dynamically-mounted Interval (inside step-content) is unreliable: Dash
# acknowledges the prop change but the browser timer may never start.

html_step_5 = html.Div(
    className="sub-progress-tracker",
    children=[

        # ── TOP SECTION: 3-column vertical stepper ───────────────────────────
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

        # ── BOTTOM SECTION: progress bar, text, button ────────────────────────
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
    Train the selected model on the full dataset for deployment.

    Delegates entirely to FinalizationRunner so that the pipeline-
    construction logic (estimator maps, two-stage/ordinal branching,
    full-data preparation) lives in one place: run_models_parallel.py.
    """
    from machine_learning.utils.training.run_models_parallel import FinalizationRunner

    runner = FinalizationRunner(
        df_data=df_data,
        df_data_surv=df_data_surv,
        model_key=model_key,
        balance_strategy=balance_strategy,
        args=args,
        best_surv_params=best_surv_params,
        fitted_cph=fitted_cph,
        use_lda=True,
    )
    pipeline, label_encoder = runner.train()

    fin_progress_state["selected_done"] = True
    return pipeline, label_encoder
 

def _train_survival_model(df_data_surv, results_root: str) -> dict:
    session          = SessionStore(results_root).load()
    survival_results = session["survival_results"]
    best_params      = survival_results["best_surv_parameters"]
    best_time_points = survival_results["best_time_points"]

    # Prepare features and structured survival target
    T = adjust_payment_period(df_data_surv["days_elapsed_until_fully_paid"])
    E = df_data_surv["censor"]
    X_raw = df_data_surv.drop(
        columns=["days_elapsed_until_fully_paid", "censor"]
    ).astype(float)

    # Apply the same standardisation as generate_survival_features._safe_scale
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    import numpy as np
    X_scaled = np.clip(X_scaled, -10, 10)

    y = Surv.from_arrays(
        event=E.astype(bool).values,
        time=T.astype(float).values,
    )

    cox = CoxnetSurvivalAnalysis(
        l1_ratio=best_params["l1_ratio"],
        alphas=[best_params["alpha"]],
        fit_baseline_model=True,   # required for S(t)/H(t) prediction in step 5
        max_iter=100_000,
        tol=1e-7,
    )
    cox.fit(X_scaled, y)
    c_index = float(cox.score(X_scaled, y))

    fin_progress_state["survival_done"] = True
    return {
        "cph":                  cox,
        "best_c_index":         c_index,
        "best_surv_parameters": best_params,
        "best_time_points":     best_time_points,
    }


def _save_pickle(obj, path: str) -> None:
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


class _FinalizationConfig:
    def __init__(self, parameters_dir, target_feature, time_points, parameters):
        self.parameters_dir = parameters_dir
        self.target_feature = target_feature
        self.time_points    = time_points
        self.parameters     = parameters


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS — STEP 5
# ══════════════════════════════════════════════════════════════════════════════

@dash_app.callback(
    Output("fin-training_status", "data"),
    Input("current_step",        "data"),
    State("selected-model-data", "data"),
    State("stored_credit_sales", "data"),
    State("stored_revenue",      "data"),
    State("stored_enrollees",    "data"),
    prevent_initial_call=True,
)
def run_finalization(current_step, selected_model_data, credit_sales_json,
                     stored_revenue, stored_enrollees):
    global _finalization_triggered

    if current_step != "progress-5":
        _finalization_triggered = False          # reset on navigation away
        return no_update

    if _finalization_triggered:
        return no_update

    if not selected_model_data:
        return "error"

    # validator that catches None, '', ' ', 'null', and malformed JSON.
    def _is_valid_dataframe_json(s) -> bool:
        if not s or not isinstance(s, str) or not s.strip():
            return False
        try:
            import json as _json
            obj = _json.loads(s)
            return isinstance(obj, dict)
        except Exception:
            return False

    # three-tier fallback when stored_credit_sales is missing/stale:
    # Tier 1 — in-memory store (normal path, always tried first via guard above)
    # Tier 2 — credit_sales_cache.pkl written to RESULTS_ROOT by step_3
    # Tier 3 — regenerate df_credit_sales from the raw uploaded file stores
    #          using the identical CreditSalesProcessor logic as step_3
    if not _is_valid_dataframe_json(credit_sales_json):
        print(
            f"[finalization] stored_credit_sales missing/unparseable "
            f"(type={type(credit_sales_json).__name__}, "
            f"preview={str(credit_sales_json)[:80] if credit_sales_json else 'None'}). "
            f"Trying disk fallback (tier 2)..."
        )
        _recovered = False

        # ── Tier 2: pkl written by step_3 ────────────────────────────────────
        try:
            import pickle as _pkl
            import os as _os
            _settings = read_settings_json()
            _cs_path  = _os.path.join(
                _settings["Training"]["RESULTS_ROOT"], "credit_sales_cache.pkl"
            )
            with open(_cs_path, "rb") as _fh:
                _df_cs = _pkl.load(_fh)
            credit_sales_json = _df_cs.to_json(date_format="iso", orient="split")
            print(f"[finalization] Tier-2 OK — {len(_df_cs)} rows from {_cs_path}")
            _recovered = True
        except Exception as _e2:
            print(f"[finalization] Tier-2 failed: {_e2}. Trying regeneration (tier 3)...")

        # ── Tier 3: regenerate from raw uploaded file stores ─────────────────
        if not _recovered:
            try:
                import base64 as _b64
                import io as _io
                from datetime import datetime as _dt
                import pandas as _pd2
                from feature_engineering.credit_sales_machine_learning import (
                    CreditSalesProcessor as _CSP,
                )

                if not stored_revenue or not stored_enrollees:
                    print("[finalization] Tier-3 failed: stored_revenue or "
                          "stored_enrollees is missing — cannot regenerate.")
                    return "error"

                _settings = read_settings_json()
                _obs_end  = _dt.strptime(
                    _settings["Training"]["observation_end"], "%Y/%m/%d"
                )

                class _Cfg:
                    observation_end = _obs_end

                _rev_bytes  = _b64.b64decode(stored_revenue.split(",")[1])
                _enr_bytes  = _b64.b64decode(stored_enrollees.split(",")[1])
                _df_rev     = _pd2.read_excel(_io.BytesIO(_rev_bytes),  engine="calamine")
                _df_enr     = _pd2.read_excel(_io.BytesIO(_enr_bytes),  engine="calamine")

                _cs = _CSP(
                    _df_rev, _df_enr, _Cfg(),
                    drop_helper_columns=True,
                    drop_demographic_columns=True,
                    drop_plan_type_columns=False,
                    drop_missing_dtp=True,
                    drop_back_account_transactions=True,
                    exclude_school_years=[2016, 2017, 2018],
                    winsorise_dtp=True,
                )
                _df_cs = _cs.show_data()
                credit_sales_json = _df_cs.to_json(date_format="iso", orient="split")
                print(f"[finalization] Tier-3 OK — regenerated {len(_df_cs)} rows "
                      "from stored_revenue / stored_enrollees.")

                # Also cache to disk so tier 2 works on the next run
                try:
                    import os as _os2, pickle as _pkl2
                    _cache = _os2.join(
                        _settings["Training"]["RESULTS_ROOT"], "credit_sales_cache.pkl"
                    )
                    _os2.makedirs(
                        _os2.dirname(_cache) if _os2.dirname(_cache) else ".", exist_ok=True
                    )
                    with open(_cache, "wb") as _fh2:
                        _pkl2.dump(_df_cs, _fh2)
                    print(f"[finalization] Cached regenerated data to {_cache}")
                except Exception as _ec:
                    print(f"[finalization] Could not cache to disk: {_ec} (non-fatal)")

                _recovered = True
            except Exception as _e3:
                print(f"[finalization] Tier-3 failed: {_e3}")

        if not _recovered:
            return "error"

    _finalization_triggered = True
    for key in ("selected_done", "survival_done", "selected_saved", "survival_saved"):
        fin_progress_state[key] = False
    fin_progress_state["start_time"] = time()

    raw_key          = selected_model_data.get("key", "")
    model_key        = raw_key.split("__")[0] if "__" in raw_key else raw_key
    balance_strategy = selected_model_data.get("strategy")
    model_parameters = selected_model_data.get("parameters", {})

    def _finalize_in_background():
        try:
            import pandas as pd

            settings        = read_settings_json()
            results_root    = settings["Training"]["RESULTS_ROOT"]
            deployed_models = settings["Training"]["DEPLOYED_MODELS"]
            os.makedirs(deployed_models, exist_ok=True)

            # credit_sales_json is now guaranteed to be a valid split-orient
            # JSON string (either from the store or loaded from disk above).
            df_credit_sales      = pd.read_json(io.StringIO(credit_sales_json), orient='split')
            survival_columns     = ['days_elapsed_until_fully_paid', 'censor']
            non_survival_columns = ['due_date', 'dtp_bracket']
            df_data      = df_credit_sales[df_credit_sales['censor'] == 1].copy()
            df_data.drop(columns=survival_columns, inplace=True)
            df_data_surv = df_credit_sales.drop(columns=non_survival_columns)

            # ── everything below is UNCHANGED from your original step_5.py ──
            survival_info    = _train_survival_model(df_data_surv, results_root)
            fitted_cph       = survival_info["cph"]
            best_surv_params = survival_info["best_surv_parameters"]
            best_time_points = survival_info["best_time_points"]

            args = _FinalizationConfig(
                parameters_dir=settings["Training"]["MODEL_PARAMETERS"],
                target_feature=settings["Training"]["target_feature"],
                time_points=best_time_points,
                parameters=model_parameters,
            )

            pipeline, label_encoder = _train_selected_model(
                df_data, df_data_surv,
                model_key, balance_strategy,
                args, best_surv_params,
                fitted_cph=fitted_cph,
            )

            import glob
            _patterns = [
                os.path.join(deployed_models, "finalized_*.pkl"),
                os.path.join(deployed_models, "finalized_survival_model.pkl"),
            ]
            for _pattern in _patterns:
                for _old_file in glob.glob(_pattern):
                    try:
                        os.remove(_old_file)
                        print(f"[finalization] Removed old model: {_old_file}")
                    except OSError as _e:
                        print(f"[finalization] Could not remove {_old_file}: {_e}")

            selected_model_path = os.path.join(deployed_models, f"finalized_{model_key}.pkl")
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

            survival_model_path = os.path.join(deployed_models, "finalized_survival_model.pkl")
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

        except Exception:
            print(f"[finalization] FATAL ERROR:\n{traceback.format_exc()}")

    threading.Thread(target=_finalize_in_background, daemon=True).start()
    return "running"


@dash_app.callback(
    Output("fin-sub-step1", "className"),
    Output("fin-sub-step2", "className"),
    Output("fin-sub-step3", "className"),
    Output("fin-sub-step4", "className"),
    Input("fin-progress-interval", "n_intervals"),
    State("fin-training_status",   "data"),
    prevent_initial_call=True,
)
def update_fin_sub_steps(n, status):
    if status != "running":
        return no_update, no_update, no_update, no_update

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


@dash_app.callback(
    Output("fin-step1-status", "children"),
    Output("fin-step2-status", "children"),
    Output("fin-step3-status", "children"),
    Output("fin-step4-status", "children"),
    Input("fin-progress-interval", "n_intervals"),
    State("fin-training_status",   "data"),
    prevent_initial_call=True,
)
def update_fin_step_statuses(n, status):
    if status != "running":
        return no_update, no_update, no_update, no_update

    step1 = "Train Selected Model."
    step2 = "Train Survival Analysis Model."
    step3 = "Save Selected Model Weights."
    step4 = "Save Survival Model Weights."

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

    return step1, step2, step3, step4


@dash_app.callback(
    Output("fin-training-progress",    "value"),
    Output("fin-training-progress",    "animated"),
    Output("fin-training-progress",    "color"),
    Output("fin-progress-text",        "children"),
    Output("finalize_btn",             "disabled"),
    Output("finalization-complete",    "data"),
    Input("fin-progress-interval",     "n_intervals"),
    State("fin-training_status",       "data"),
    prevent_initial_call=True,
)
def update_fin_progress(n, status):
    if status != "running":
        return no_update, no_update, no_update, no_update, no_update, no_update

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
        return 100, False, "success", "Finalization complete. Model is ready.", False, True

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
        return percent, True, "primary", f"{completed}/{total} steps completed ({percent}%) — {eta_str}", True, no_update

    return 0, True, "primary", "Waiting for finalization to start...", True, no_update