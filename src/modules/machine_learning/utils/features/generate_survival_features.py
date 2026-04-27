import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from scipy.integrate import trapezoid

from src.modules.machine_learning.utils.data.clean_survival_inputs import clean_survival_inputs


# exp() overflows float64 at ~709.  Clipping standardised columns to
# [-_COL_CLIP, +_COL_CLIP] keeps the linear predictor finite.
_COL_CLIP = 10.0


def _safe_scale(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardise df column-wise, clip to [-_COL_CLIP, +_COL_CLIP].

    If a fitted scaler is supplied it is reused (apply train stats to test
    without leakage).  Returns (scaled_df, scaler).
    """
    if scaler is None:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df)
    else:
        scaled_values = scaler.transform(df)

    scaled = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)
    scaled = scaled.clip(lower=-_COL_CLIP, upper=_COL_CLIP)
    return scaled, scaler


def _sanitize(df: pd.DataFrame, ref_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Replace inf/-inf with NaN then impute with column medians.
    Uses ref_df medians for test data to prevent leakage.
    Falls back to 0 if the median is itself non-finite.
    """
    df     = df.replace([np.inf, -np.inf], np.nan)
    source = ref_df if ref_df is not None else df
    medians = source.median().replace([np.inf, -np.inf], np.nan).fillna(0)
    return df.fillna(medians)


def _expected_survival_time(sf) -> float:
    """
    Compute E[T] = integral_0^inf S(t) dt from a sksurv StepFunction
    using the trapezoidal rule.
    """
    t = sf.x
    s = sf(t)
    if len(t) == 0:
        return 0.0
    # Prepend t=0, S(0)=1 if missing
    if t[0] > 0:
        t = np.concatenate([[0.0], t])
        s = np.concatenate([[1.0], s])
    return float(trapezoid(s, t))


def _compute_survival_features(
    cox: CoxnetSurvivalAnalysis,
    X_raw: pd.DataFrame,
    scaler: StandardScaler,
    safe_time_points: list,
) -> pd.DataFrame:
    """
    Apply scaled CoxnetSurvivalAnalysis predictions to one feature matrix.
    Shared by both train and test paths.
    """
    X_scaled, _ = _safe_scale(X_raw, scaler=scaler)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # Survival probabilities S(t) at each time point
        surv_fns = cox.predict_survival_function(X_scaled)
        surv_probs = pd.DataFrame(
            {f"surv_prob_{t}": np.array([fn(t) for fn in surv_fns])
             for t in safe_time_points},
            index=X_raw.index,
        )

        # Cumulative hazards H(t) at each time point
        chf_fns = cox.predict_cumulative_hazard_function(X_scaled)
        cum_hazards = pd.DataFrame(
            {f"cum_hazard_{t}": np.array([fn(t) for fn in chf_fns])
             for t in safe_time_points},
            index=X_raw.index,
        )

        # Risk scores: cox.predict() returns the linear predictor (log hazard ratio)
        log_risk      = cox.predict(X_scaled)
        partial_hazard = np.exp(log_risk).clip(min=1e-6)

        # Expected survival time: integrate S(t) over observed time grid
        expected_times = np.array([_expected_survival_time(fn) for fn in surv_fns])

    survival_df = pd.DataFrame({
        "expected_survival_time": expected_times,
        "partial_hazard":         partial_hazard,
        "log_partial_hazard":     np.log(partial_hazard),
    }, index=X_raw.index)

    result = pd.concat([
        X_raw.reset_index(drop=True),
        surv_probs.reset_index(drop=True),
        cum_hazards.reset_index(drop=True),
        survival_df.reset_index(drop=True),
    ], axis=1)

    # Final clip to prevent float32 overflow inside sklearn
    numeric_cols = result.select_dtypes(include=np.number).columns
    result[numeric_cols] = result[numeric_cols].clip(lower=-1e15, upper=1e15)

    return result


def generate_survival_features(
    X_surv,
    T,
    E,
    X_train: Optional[pd.DataFrame],
    X_test: Optional[pd.DataFrame] = None,
    best_params: dict = None,
    time_points: list = None,
    fitted_cph=None,
    cox_scaler=None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Fit (or reuse) a CoxnetSurvivalAnalysis model and generate
    survival-informed features for downstream classifiers.

    Parameters
    ----------
    X_surv : pd.DataFrame
        Feature matrix used to fit the Cox model (ignored when fitted_cph
        is supplied).
    T : array-like
        Time-to-event values. Must be strictly positive.
    E : array-like
        Event indicator (1 = event observed, 0 = censored).
    X_train : pd.DataFrame
        Training features for the downstream classifier.
    X_test : pd.DataFrame, optional
        Test features.  When provided, transformed using train scaler and
        returned as the second element of the output tuple.
    best_params : dict, optional
        Output of CoxHyperparameterTuner.  Required when fitted_cph is None.
        Expected keys: "alpha", "l1_ratio".
    time_points : list of int, optional
        Times at which to compute S(t) and H(t).  Defaults to [30, 60, 90, 120].
    fitted_cph : CoxnetSurvivalAnalysis, optional
        A pre-fitted model.  When supplied no new fitting is performed.
        The parameter name "fitted_cph" is kept for backward compatibility.
    cox_scaler : StandardScaler, optional
        A pre-fitted scaler from the Cox model fitting.  Used with fitted_cph
        to avoid refitting the scaler. When supplied with fitted_cph, skips
        scaler refitting.

    Returns
    -------
    df_train_survival : pd.DataFrame
    df_test_survival : pd.DataFrame   (only when X_test is not None)

    Raises
    ------
    ValueError
        If inf/NaN remain after sanitization, or if neither best_params
        nor fitted_cph is supplied.
    """
    if time_points is None:
        time_points = [30, 60, 90, 120]

    # ── Step 1: fit or reuse Cox model ───────────────────────────────────────
    if fitted_cph is not None and cox_scaler is not None:
        # Reuse caller-supplied model AND scaler — skip all refitting
        cox = fitted_cph
        fit_scaler = cox_scaler

    elif fitted_cph is not None:
        # Reuse caller-supplied model — derive scaler from reference data.
        cox = fitted_cph
        _ref = X_surv if X_surv is not None else X_train
        _, _, _, df_fit = clean_survival_inputs(_ref, T, E)
        feature_cols = [c for c in df_fit.columns if c not in ("T", "E")]
        _, fit_scaler = _safe_scale(df_fit[feature_cols])

    else:
        if best_params is None:
            raise ValueError("Either best_params or fitted_cph must be supplied.")

        _, _, _, df_fit = clean_survival_inputs(X_surv, T, E)
        feature_cols = [c for c in df_fit.columns if c not in ("T", "E")]
        X_fit_raw    = df_fit[feature_cols]

        X_fit_scaled, fit_scaler = _safe_scale(X_fit_raw)

        y_fit = Surv.from_arrays(
            event=df_fit["E"].astype(bool).values,
            time=df_fit["T"].astype(float).values,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cox = CoxnetSurvivalAnalysis(
                l1_ratio=best_params["l1_ratio"],
                alphas=[best_params["alpha"]],
                fit_baseline_model=True,   # required for S(t) and H(t) prediction
                max_iter=100_000,
                tol=1e-7,
            )
            cox.fit(X_fit_scaled, y_fit)

    # Replace t <= 0 with t=1 (S(0) is undefined)
    safe_time_points = []
    replaced = []
    for t in time_points:
        if t <= 0:
            safe_time_points.append(1)
            replaced.append(t)
        else:
            safe_time_points.append(t)

    if replaced:
        print(
            f"[generate_survival_features] WARNING: Replaced t={set(replaced)} with t=1 — "
            f"survival functions are undefined at t<=0."
        )

    # ── Step 2: generate features ─────────────────────────────────────────────
    df_train_survival = _sanitize(
        _compute_survival_features(cox, X_train, fit_scaler, safe_time_points)
    )

    # ── Step 3: optional test set ─────────────────────────────────────────────
    if X_test is not None:
        df_test_survival = _sanitize(
            _compute_survival_features(cox, X_test, fit_scaler, safe_time_points),
            ref_df=df_train_survival,
        )
    else:
        df_test_survival = None

    # ── Step 4: final guard ───────────────────────────────────────────────────
    datasets = [("TRAIN", df_train_survival)]
    if df_test_survival is not None:
        datasets.append(("TEST", df_test_survival))

    for name, df in datasets:
        n_inf = np.isinf(df.select_dtypes(include=np.number).values).sum()
        n_nan = df.isnull().sum().sum()
        if n_inf > 0 or n_nan > 0:
            raise ValueError(
                f"[generate_survival_features] {name} still contains "
                f"{n_inf} inf and {n_nan} NaN values after sanitization."
            )

    if df_test_survival is not None:
        return df_train_survival, df_test_survival

    return df_train_survival