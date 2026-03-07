import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

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


def generate_survival_features(
    X_surv, T, E, X_train, X_test,
    best_params,
    time_points=[30, 60, 90, 120]
):
    """
    Fit a CoxPHFitter and generate survival-informed features for downstream
    classifiers (e.g. AdaBoost).

    Overflow prevention strategy
    ----------------------------
    The root cause of exp() overflow is an unbounded linear predictor LP = X @ beta.
    This can occur both during fitting (Newton-Raphson gradient steps) and during
    prediction. The fix is applied at the data level before anything reaches lifelines:

    1. All feature columns are standardised (zero mean, unit variance) and then
       hard-clipped to [-10, 10] z-scores. With typical penalised Cox betas well
       below 1.0, this keeps |LP| comfortably below 709 (float64 exp() limit).
    2. Train statistics (mean, std) are computed once and reused for X_test and
       X_train to prevent leakage.
    3. Residual inf/NaN in outputs are imputed with train-set medians.
    4. A final guard raises ValueError if any non-finite values survive.

    Parameters
    ----------
    X_surv : pd.DataFrame
        Feature matrix used to fit the Cox model.
    T : array-like
        Time-to-event values. Must be strictly positive.
    E : array-like
        Event indicator (1=event observed, 0=censored).
    X_train : pd.DataFrame
        Training features for the downstream classifier.
    X_test : pd.DataFrame
        Test features. Sanitized using train statistics.
    best_params : dict
        Output of tune_cox_hyperparameters(). Keys: 'penalizer', 'l1_ratio',
        'baseline_estimation_method', 'robust', 'step_size'.
    time_points : list of int
        Times at which to compute S(t) and H(t). t=0 is automatically removed.

    Returns
    -------
    df_train_survival : pd.DataFrame
        Shape: (n_train, n_features + 2*len(safe_time_points) + 3)
    df_test_survival : pd.DataFrame
        Shape: (n_test,  n_features + 2*len(safe_time_points) + 3)

    Raises
    ------
    ValueError
        If inf or NaN values remain after sanitization.
    """
    _, _, _, df_fit = clean_survival_inputs(X_surv, T, E)

    # ── Step 1: standardise + clip fitting data to suppress exp() overflow ──
    feature_cols = [c for c in df_fit.columns if c not in ("T", "E")]
    X_fit_raw = df_fit[feature_cols]
    X_fit_scaled, fit_mean, fit_std = _safe_scale(X_fit_raw)

    df_fit_scaled = pd.concat(
        [X_fit_scaled, df_fit[["T", "E"]].reset_index(drop=True)],
        axis=1
    )

    # ── Step 2: fit the Cox model on the clipped, scaled data ──
    cph = CoxPHFitter(
        penalizer=best_params["penalizer"],
        l1_ratio=best_params["l1_ratio"],
        baseline_estimation_method=best_params["baseline_estimation_method"]
    )

    # Suppress the RuntimeWarnings that lifelines emits mid-optimisation;
    # the solver recovers from these internally via step-halving.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cph.fit(
            df_fit_scaled,
            duration_col="T",
            event_col="E",
            robust=best_params["robust"],
            fit_options={"step_size": best_params["step_size"]}
        )

    # Remove t=0: S(0) is undefined and causes H(0) = -log(0) = inf
    safe_time_points = [t for t in time_points if t > 0]
    if len(safe_time_points) < len(time_points):
        removed = set(time_points) - set(safe_time_points)
        print(f"[generate_survival_features] WARNING: Removed t={removed} — "
              f"survival functions are undefined at t=0.")

    def _compute_features(X_raw, col_mean, col_std):
        # Apply the same scaling used during fit (train stats → no leakage)
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
            "partial_hazard": partial_hazard,
            "log_partial_hazard": np.log(partial_hazard)
        }, index=X_raw.index)

        result = pd.concat([
            X_raw.reset_index(drop=True),
            surv_probs.reset_index(drop=True),
            cum_hazards.reset_index(drop=True),
            survival_df.reset_index(drop=True)
        ], axis=1)

        # Clip extreme finite values to prevent float32 overflow inside sklearn
        numeric_cols = result.select_dtypes(include=np.number).columns
        result[numeric_cols] = result[numeric_cols].clip(lower=-1e15, upper=1e15)

        return result

    # ── Step 3: generate features, impute residual non-finites ──
    df_train_survival = _sanitize(_compute_features(X_train, fit_mean, fit_std))
    df_test_survival  = _sanitize(
        _compute_features(X_test, fit_mean, fit_std),
        ref_df=df_train_survival
    )

    # ── Step 4: final guard ──
    for name, df in [("TRAIN", df_train_survival), ("TEST", df_test_survival)]:
        n_inf = np.isinf(df.select_dtypes(include=np.number).values).sum()
        n_nan = df.isnull().sum().sum()
        if n_inf > 0 or n_nan > 0:
            raise ValueError(
                f"[generate_survival_features] {name} still contains "
                f"{n_inf} inf and {n_nan} NaN values after sanitization."
            )

    return df_train_survival, df_test_survival