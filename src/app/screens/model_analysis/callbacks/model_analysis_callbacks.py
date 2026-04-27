# app/callbacks/model_analysis_callbacks.py
#
# Screen 2 — Standalone Analysis.
# Imports shared callback modules (core + filters are already registered by
# Screen 1's import chain; guard against double-registration is handled by
# Python's module cache — each module only executes once per process).
# Adds the session-selector callbacks that are unique to this screen.
# Also handles navigation from the main dashboard's "View Saved Models" button.

import src.app.screens.comparative_model_dashboard_template.callbacks.core     # no-op if already imported
import src.app.screens.comparative_model_dashboard_template.callbacks.filters  # no-op if already imported
import src.app.screens.comparative_model_dashboard_template.callbacks.screen_2 # session dropdown + on-demand load

from src.app.screens.comparative_model_dashboard_template.dashboard_layout import build_dashboard_layout
from dash import Input, Output, no_update
from dash import dcc

from src.app import dash_app

# Consumed by model_analysis.py → ModelAnalysisScreen.layout()
html_analysis = build_dashboard_layout(show_session_selector=True)


# ── Navigate to Model Analysis (Screen 2) ─────────────────────────────────────

@dash_app.callback(
    Output("url", "pathname"),
    Input("view_models_btn", "n_clicks"),
    prevent_initial_call=True,
)
def navigate_to_model_analysis(n_clicks):
    """
    Fires when the user clicks "View Saved Models" on the main dashboard.
    Updates the URL to /analysis, which the router in app.py maps to
    ModelAnalysisScreen.
    """
    if not n_clicks:
        return no_update
    return "/analysis"


# ── Inference helper (Fit-Transform pattern) ──────────────────────────────────
#
# When predicting on NEW invoice data, the feature engineering step must use
# the SAME plan_type_risk_score mapping that was computed during training.
# The mapping is embedded inside the InferencePipeline pickle as:
#
#     inf.feature_metadata["plan_risk_map"]
#
# Usage pattern — call load_inference_pipeline() before preparing features:
#
#     inf  = load_inference_pipeline(pkl_path)
#     proba = predict_on_new_data(inf, df_revenues_new, df_enrollees_new, args)
#
# This guarantees that even a single-row inference batch receives the same
# risk score a plan type would have received at training time.

def load_inference_pipeline(pkl_path: str):
    """
    Load a finalized InferencePipeline from disk.

    Parameters
    ----------
    pkl_path : str
        Absolute path to the ``finalized_*.pkl`` file written by step_5.

    Returns
    -------
    InferencePipeline
        The fully self-contained bundle (scaler, Cox model, classifier,
        label encoder, LDA transformer, and feature_metadata).
    """
    import pickle
    with open(pkl_path, "rb") as fh:
        return pickle.load(fh)


def predict_on_new_data(inf, df_revenues_new, df_enrollees_new, args):
    """
    Prepare raw invoice data and predict payment delay probabilities.

    Implements the Transform step of the Fit-Transform pattern:
    the ``plan_risk_map`` stored inside ``inf.feature_metadata`` is passed
    to ``CreditSalesProcessor`` so that new data is scored using the
    training-set distribution regardless of the batch size.

    Parameters
    ----------
    inf : InferencePipeline
        Loaded via ``load_inference_pipeline``.
    df_revenues_new : pd.DataFrame
        Raw revenue records for the new batch (same schema as training data).
    df_enrollees_new : pd.DataFrame
        Enrollment records for the new batch.
    args : object
        Must expose ``observation_end`` (datetime).

    Returns
    -------
    pd.DataFrame
        Columns are the class labels (e.g. on_time, 30_days, 60_days, 90_days);
        rows are probabilities for each invoice.
    """
    from src.modules.feature_engineering.credit_sales_machine_learning import CreditSalesProcessor

    # Extract the training-set plan_risk_map (may be None for legacy pkls).
    plan_risk_map = inf.feature_metadata.get("plan_risk_map")

    cs = CreditSalesProcessor(
        df_revenues_new,
        df_enrollees_new,
        args,
        drop_helper_columns=True,
        drop_demographic_columns=True,
        drop_plan_type_columns=False,
        drop_missing_dtp=False,   # keep uncensored rows for prediction
        drop_back_account_transactions=True,
        winsorise_dtp=True,
        plan_risk_map=plan_risk_map,
    )
    X_raw = cs.show_data()

    # Drop survival / label columns that are not features.
    cols_to_drop = [c for c in ["days_elapsed_until_fully_paid", "censor",
                                 "due_date", "dtp_bracket"] if c in X_raw.columns]
    X_raw = X_raw.drop(columns=cols_to_drop)

    return inf.predict(X_raw)
