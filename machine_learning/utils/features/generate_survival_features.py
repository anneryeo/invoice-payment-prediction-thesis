import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from machine_learning.utils.training.clean_survival_inputs import clean_survival_inputs


def generate_survival_features(
    X_surv, T, E, X_train, X_test,
    best_params,
    time_points=[30, 60, 90, 120]
):
    """
    Fit a CoxPHFitter and generate survival-informed features for downstream
    classifiers (e.g. AdaBoost).

    The Cox model is fitted on (X_surv, T, E) and used to extract four feature
    types per observation: survival probabilities S(t), cumulative hazards H(t),
    expected survival time E[T|X], and partial hazard scores. These capture
    time-to-event dynamics that classical classifiers cannot learn from raw
    features alone.

    Non-finite values (inf from H(t)->inf as S(t)->0, -inf from log of near-zero
    hazards) are replaced with training-set medians. Test imputation uses train
    medians to prevent leakage. t=0 is explicitly removed from time_points as
    survival functions are undefined/unstable at t=0. Extreme finite values are
    clipped to [-1e15, 1e15] to prevent float32 overflow inside sklearn.

    Parameters
    ----------
    X_surv : pd.DataFrame
        Feature matrix used to fit the Cox model. Should not overlap with
        X_test to avoid leakage into test predictions.
    T : array-like
        Time-to-event values. Must be strictly positive.
    E : array-like
        Event indicator (1=event observed, 0=censored).
    X_train : pd.DataFrame
        Training features for the downstream classifier.
    X_test : pd.DataFrame
        Test features. Sanitized using train medians.
    best_params : dict
        Output of tune_cox_hyperparameters(). Keys: 'penalizer', 'l1_ratio',
        'baseline_estimation_method', 'robust', 'step_size'.
    time_points : list of int
        Times at which to compute S(t) and H(t). t=0 is automatically removed.
        Use get_slope_timepoints() for data-driven selection anchored to
        high-variance regions of the KM curve.

    Returns
    -------
    df_train_survival : pd.DataFrame
        Shape: (n_train, n_features + 2*len(safe_time_points) + 3)
    df_test_survival : pd.DataFrame
        Shape: (n_test, n_features + 2*len(safe_time_points) + 3)

    Raises
    ------
    ValueError
        If inf or NaN values remain after sanitization.
    """
    _, _, _, df_fit = clean_survival_inputs(X_surv, T, E)

    cph = CoxPHFitter(
        penalizer=best_params["penalizer"],
        l1_ratio=best_params["l1_ratio"],
        baseline_estimation_method=best_params["baseline_estimation_method"]
    )
    cph.fit(
        df_fit,
        duration_col="T",
        event_col="E",
        robust=best_params["robust"],
        fit_options={"step_size": best_params["step_size"]}
    )

    # Remove t=0: S(0) is undefined and causes H(0) = -log(0) = inf
    safe_time_points = [t for t in time_points if t > 0]
    if len(safe_time_points) < len(time_points):
        removed = set(time_points) - set(safe_time_points)
        print(f"[generate_survival_features] WARNING: Removed t={removed} from "
              f"time_points. Survival functions are undefined at t=0.")

    def _sanitize(df, ref_df=None):
        """
        Replace inf/-inf with NaN, impute with median.
        Falls back to 0 if median is itself NaN/inf (e.g. entire column was bad).
        Uses ref_df medians for test set to prevent leakage.
        """
        df = df.replace([np.inf, -np.inf], np.nan)
        source = ref_df if ref_df is not None else df
        medians = source.median()
        # Guard: if median is inf/nan (>50% of column was non-finite), fall back to 0
        medians = medians.replace([np.inf, -np.inf], np.nan).fillna(0)
        df = df.fillna(medians)
        return df

    def _compute_features(X):
        surv_probs = pd.DataFrame({
            f"surv_prob_{t}": cph.predict_survival_function(X, times=[t]).T.squeeze()
            for t in safe_time_points
        }, index=X.index)

        cum_hazards = pd.DataFrame({
            f"cum_hazard_{t}": cph.predict_cumulative_hazard(X, times=[t]).T.squeeze()
            for t in safe_time_points
        }, index=X.index)

        expected_survival_time = cph.predict_expectation(X)

        # Clip before log to prevent log(0) -> -inf
        partial_hazard = cph.predict_partial_hazard(X).clip(lower=1e-6)

        survival_df = pd.DataFrame({
            "expected_survival_time": expected_survival_time,
            "partial_hazard": partial_hazard,
            "log_partial_hazard": np.log(partial_hazard)
        }, index=X.index)

        result = pd.concat([
            X.reset_index(drop=True),
            surv_probs.reset_index(drop=True),
            cum_hazards.reset_index(drop=True),
            survival_df.reset_index(drop=True)
        ], axis=1)

        # Clip extreme finite values to prevent float32 overflow inside sklearn
        # float32 max ~3.4e38 but sklearn casts conservatively — 1e15 is safe
        numeric_cols = result.select_dtypes(include=np.number).columns
        result[numeric_cols] = result[numeric_cols].clip(lower=-1e15, upper=1e15)

        return result

    df_train_survival = _sanitize(_compute_features(X_train))

    # Use train medians for test imputation to avoid data leakage
    df_test_survival = _sanitize(_compute_features(X_test), ref_df=df_train_survival)

    # Final guard: catch anything that survived sanitization
    for name, df in [("TRAIN", df_train_survival), ("TEST", df_test_survival)]:
        n_inf = np.isinf(df.select_dtypes(include=np.number).values).sum()
        n_nan = df.isnull().sum().sum()
        if n_inf > 0 or n_nan > 0:
            raise ValueError(
                f"[generate_survival_features] {name} still contains "
                f"{n_inf} inf and {n_nan} NaN values after sanitization."
            )

    return df_train_survival, df_test_survival