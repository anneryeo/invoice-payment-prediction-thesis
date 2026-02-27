import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

def generate_survival_features(X, T, E, best_penalty, time_points=[30, 60, 90, 120]):
    """
    Generate survival-based features using CoxPHFitter from lifelines.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    T : array-like
        Time-to-event values.
    E : array-like
        Event indicator (1=event occurred, 0=censored).
    best_penalty : float
        Penalizer value chosen from tuning.
    time_points : list of int
        Time points at which to compute survival probabilities and cumulative hazards.

    Returns
    -------
    df_survival : pd.DataFrame
        Extended feature matrix with survival-based features.
    """

    # Fit Cox model via lifelines
    cph = CoxPHFitter(penalizer=best_penalty)
    df_fit = pd.concat([X, pd.DataFrame({"T": T, "E": E})], axis=1)
    cph.fit(df_fit, duration_col="T", event_col="E")

    # Survival probabilities at specified time points
    surv_probs = pd.DataFrame({
        f"surv_prob_{t}": cph.predict_survival_function(X, times=[t]).T.squeeze()
        for t in time_points
    })

    # Cumulative hazards at specified time points
    cum_hazards = pd.DataFrame({
        f"cum_hazard_{t}": cph.predict_cumulative_hazard(X, times=[t]).T.squeeze()
        for t in time_points
    })

    # Expected survival time and partial hazard
    expected_survival_time = cph.predict_expectation(X)
    partial_hazard = cph.predict_partial_hazard(X)

    # Combine everything
    df_survival = pd.concat([
        X,
        surv_probs,
        cum_hazards,
        pd.DataFrame({
            "expected_survival_time": expected_survival_time,
            "partial_hazard": partial_hazard,
            "log_partial_hazard": np.log1p(partial_hazard)
        })
    ], axis=1)

    return df_survival