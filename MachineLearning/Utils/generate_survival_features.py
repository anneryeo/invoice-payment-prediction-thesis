import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

def generate_survival_features(X_surv, T, E, X_train, X_test, best_penalty, time_points=[30, 60, 90, 120]):
    """
    Generate survival-based features using CoxPHFitter from lifelines.

    Parameters
    ----------
    X_surv : pd.DataFrame
        Feature matrix used for fitting the Cox model.
    T : array-like
        Time-to-event values.
    E : array-like
        Event indicator (1=event occurred, 0=censored).
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Test feature matrix.
    best_penalty : float
        Penalizer value chosen from tuning.
    time_points : list of int
        Time points at which to compute survival probabilities and cumulative hazards.

    Returns
    -------
    df_train_survival : pd.DataFrame
        Extended training feature matrix with survival-based features.
    df_test_survival : pd.DataFrame
        Extended test feature matrix with survival-based features.
    """
    # Fit Cox model
    cph = CoxPHFitter(penalizer=best_penalty)
    df_fit = pd.concat([X_surv, pd.DataFrame({"T": T, "E": E})], axis=1)
    cph.fit(df_fit, duration_col="T", event_col="E")

    def _compute_features(X):
        # Survival probabilities
        surv_probs = pd.DataFrame({
            f"surv_prob_{t}": cph.predict_survival_function(X, times=[t]).T.squeeze()
            for t in time_points
        })

        # Cumulative hazards
        cum_hazards = pd.DataFrame({
            f"cum_hazard_{t}": cph.predict_cumulative_hazard(X, times=[t]).T.squeeze()
            for t in time_points
        })

        # Expected survival time and partial hazard
        expected_survival_time = cph.predict_expectation(X)
        partial_hazard = cph.predict_partial_hazard(X)

        # Combine everything
        df_survival = pd.concat([
            X.reset_index(drop=True),
            surv_probs.reset_index(drop=True),
            cum_hazards.reset_index(drop=True),
            pd.DataFrame({
                "expected_survival_time": expected_survival_time,
                "partial_hazard": partial_hazard,
                "log_partial_hazard": np.log1p(partial_hazard)
            }).reset_index(drop=True)
        ], axis=1)

        return df_survival

    # Compute features for train and test
    df_train_survival = _compute_features(X_train)
    df_test_survival = _compute_features(X_test)

    return df_train_survival, df_test_survival