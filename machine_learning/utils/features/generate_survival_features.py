import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from typing import Optional, Tuple, Union

from machine_learning.utils.data.clean_survival_inputs import clean_survival_inputs


# exp() overflows float64 at ~709. With p features, each contributing
# beta_i * x_i, we budget LP_CLIP / p per feature after standardisation.
# Clipping each standardised column to [-COL_CLIP, +COL_CLIP] ensures the
# sum stays finite regardless of how many features there are.
_COL_CLIP = 10.0   # z-score clip applied to every column before fit & predict


def _safe_scale(df: pd.DataFrame, mean: pd.Series = None, std: pd.Series = None):
    """
    Standardise df column-wise then clip to [-_COL_CLIP, +_COL_CLIP].

    If mean/std are supplied they are reused (for applying train statistics to
    test data without leakage). Returns (scaled_df, mean, std).
    """
    if mean is None:
        mean = df.mean()
    if std is None:
        std = df.std().replace(0, 1)   # avoid /0 for constant columns

    scaled = (df - mean) / std
    scaled = scaled.clip(lower=-_COL_CLIP, upper=_COL_CLIP)
    return scaled, mean, std


def _sanitize(df: pd.DataFrame, ref_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Replace inf/-inf with NaN then impute with column medians.
    Uses ref_df medians for test data to prevent leakage.
    Falls back to 0 if the median is itself non-finite.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    source = ref_df if ref_df is not None else df
    medians = source.median().replace([np.inf, -np.inf], np.nan).fillna(0)
    return df.fillna(medians)


def _compute_survival_features(
    cph: CoxPHFitter,
    X_raw: pd.DataFrame,
    col_mean: pd.Series,
    col_std: pd.Series,
    safe_time_points: list,
) -> pd.DataFrame:
    """
    Internal helper: apply scaled Cox predictions to a single feature matrix.
    Extracted so both train-only and train+test paths share identical logic.
    """
    X_scaled, _, _ = _safe_scale(X_raw, mean=col_mean, std=col_std)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        surv_probs = pd.DataFrame({
            f"surv_prob_{t}": cph.predict_survival_function(
                X_scaled, times=[t]
            ).T.squeeze()
            for t in safe_time_points
        }, index=X_raw.index)

        cum_hazards = pd.DataFrame({
            f"cum_hazard_{t}": cph.predict_cumulative_hazard(
                X_scaled, times=[t]
            ).T.squeeze()
            for t in safe_time_points
        }, index=X_raw.index)

        expected_survival_time = cph.predict_expectation(X_scaled)
        partial_hazard = cph.predict_partial_hazard(X_scaled).clip(lower=1e-6)

    survival_df = pd.DataFrame({
        "expected_survival_time": expected_survival_time,
        "partial_hazard":         partial_hazard,
        "log_partial_hazard":     np.log(partial_hazard),
    }, index=X_raw.index)

    result = pd.concat([
        X_raw.reset_index(drop=True),
        surv_probs.reset_index(drop=True),
        cum_hazards.reset_index(drop=True),
        survival_df.reset_index(drop=True),
    ], axis=1)

    # Clip extreme finite values to prevent float32 overflow inside sklearn
    numeric_cols = result.select_dtypes(include=np.number).columns
    result[numeric_cols] = result[numeric_cols].clip(lower=-1e15, upper=1e15)

    return result


def generate_survival_features(
    X_surv,
    T,
    E,
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    best_params: dict = None,
    time_points: list = None,
    fitted_cph: Optional[CoxPHFitter] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Fit (or reuse) a CoxPHFitter and generate survival-informed features for
    downstream classifiers.

    X_test is **optional**.  When omitted the function returns only
    ``df_train_survival`` as a single DataFrame.  When provided it returns the
    familiar ``(df_train_survival, df_test_survival)`` tuple.

    Reusing a pre-fitted model
    --------------------------
    Pass a ``CoxPHFitter`` instance via ``fitted_cph`` to skip fitting entirely
    and use the model from a previous training run (e.g. Step 3's saved Cox
    model).  When ``fitted_cph`` is supplied, ``X_surv``, ``T``, ``E``, and
    ``best_params`` are ignored for fitting purposes — they are only used to
    derive ``safe_time_points`` when ``time_points`` is also ``None``.

    Overflow prevention strategy
    ----------------------------
    The root cause of exp() overflow is an unbounded linear predictor
    LP = X @ beta.  The fix is applied at the data level:

    1. All feature columns are standardised (zero mean, unit variance) and
       hard-clipped to [-10, 10] z-scores.
    2. Train statistics (mean, std) are computed once and reused for X_test to
       prevent leakage.
    3. Residual inf/NaN in outputs are imputed with train-set medians.
    4. A final guard raises ValueError if any non-finite values survive.

    Parameters
    ----------
    X_surv : pd.DataFrame
        Feature matrix used to fit the Cox model (ignored when fitted_cph is
        supplied).
    T : array-like
        Time-to-event values. Must be strictly positive.
    E : array-like
        Event indicator (1=event observed, 0=censored).
    X_train : pd.DataFrame
        Training features for the downstream classifier.
    X_test : pd.DataFrame, optional
        Test features. When provided, sanitized using train statistics and
        returned as the second element of the output tuple.
    best_params : dict, optional
        Output of tune_cox_hyperparameters(). Required when fitted_cph is None.
        Keys: 'penalizer', 'l1_ratio', 'baseline_estimation_method', 'robust',
        'step_size'.
    time_points : list of int, optional
        Times at which to compute S(t) and H(t). Defaults to
        [30, 60, 90, 120].  t=0 is automatically removed.
    fitted_cph : CoxPHFitter, optional
        A pre-fitted CoxPHFitter instance.  When supplied the model is reused
        as-is and no new fitting is performed.

    Returns
    -------
    df_train_survival : pd.DataFrame
        Shape (n_train, n_features + 2*len(safe_time_points) + 3).
    df_test_survival : pd.DataFrame  *(only when X_test is not None)*
        Shape (n_test, n_features + 2*len(safe_time_points) + 3).

    Raises
    ------
    ValueError
        If inf or NaN values remain after sanitization, or if neither
        best_params nor fitted_cph is provided.
    """
    if time_points is None:
        time_points = [30, 60, 90, 120]

    # ── Step 1: fit or reuse Cox model ───────────────────────────────────────
    if fitted_cph is not None:
        # Reuse the caller-supplied model — no fitting needed.
        cph = fitted_cph

        # We still need fit_mean / fit_std for safe_scale.  Derive them from
        # X_surv if available; otherwise fall back to X_train statistics so
        # the function works even when X_surv is None.
        _ref = X_surv if X_surv is not None else X_train
        _, _, _, df_fit = clean_survival_inputs(_ref, T, E)
        feature_cols = [c for c in df_fit.columns if c not in ("T", "E")]
        _, fit_mean, fit_std = _safe_scale(df_fit[feature_cols])

    else:
        # Fresh fit path — identical to the original implementation.
        if best_params is None:
            raise ValueError(
                "Either best_params or fitted_cph must be supplied."
            )

        _, _, _, df_fit = clean_survival_inputs(X_surv, T, E)
        feature_cols = [c for c in df_fit.columns if c not in ("T", "E")]
        X_fit_raw = df_fit[feature_cols]
        X_fit_scaled, fit_mean, fit_std = _safe_scale(X_fit_raw)

        df_fit_scaled = pd.concat(
            [X_fit_scaled, df_fit[["T", "E"]].reset_index(drop=True)],
            axis=1,
        )

        cph = CoxPHFitter(
            penalizer=best_params["penalizer"],
            l1_ratio=best_params["l1_ratio"],
            baseline_estimation_method=best_params["baseline_estimation_method"],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cph.fit(
                df_fit_scaled,
                duration_col="T",
                event_col="E",
                robust=best_params["robust"],
                fit_options={"step_size": best_params["step_size"]},
            )

    # Replace t=0 to t=1: S(0) is undefined and causes H(0) = -log(0) = inf
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
            f"survival functions are undefined at t≤0. t=1 is the earliest meaningful time point."
        )

    # ── Step 2: generate features, impute residual non-finites ───────────────
    df_train_survival = _sanitize(
        _compute_survival_features(cph, X_train, fit_mean, fit_std, safe_time_points)
    )

    # ── Step 3: optional test set ─────────────────────────────────────────────
    if X_test is not None:
        df_test_survival = _sanitize(
            _compute_survival_features(cph, X_test, fit_mean, fit_std, safe_time_points),
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