import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def clean_survival_inputs(X_surv, T, E):
    """
    Internal helper: resets indexes, removes rows with NaN/inf in T or E,
    scales features, and returns a clean df_fit ready for CoxPHFitter.

    Returns
    -------
    X_surv_scaled : pd.DataFrame
    T : pd.Series
    E : pd.Series
    df_fit : pd.DataFrame
    """
    T = pd.Series(T).reset_index(drop=True)
    E = pd.Series(E).reset_index(drop=True)
    X_surv = X_surv.reset_index(drop=True)

    # Drop rows where T or E are NaN or non-finite
    valid_mask = (
        T.notna() & E.notna() &
        np.isfinite(T.astype(float)) & np.isfinite(E.astype(float))
    )
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        print(f"[clean_survival_inputs] Dropping {n_dropped} rows with NaN/inf in T or E.")

    X_surv = X_surv[valid_mask].reset_index(drop=True)
    T = T[valid_mask].reset_index(drop=True)
    E = E[valid_mask].reset_index(drop=True)

    # Scale predictors; fill NaNs from zero-variance columns
    scaler = StandardScaler()
    X_surv_scaled = pd.DataFrame(scaler.fit_transform(X_surv), columns=X_surv.columns)
    X_surv_scaled = X_surv_scaled.fillna(0.0)

    # Build and clean df_fit
    df_fit = pd.concat([X_surv_scaled, pd.DataFrame({"T": T, "E": E})], axis=1)
    df_fit = df_fit.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if df_fit.empty:
        raise ValueError(
            "[clean_survival_inputs] df_fit is empty after NaN/inf removal. "
            "Check your T and E arrays for missing or invalid values."
        )

    return X_surv_scaled, T, E, df_fit