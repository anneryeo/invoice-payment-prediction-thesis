import numpy as np
import pandas as pd
from itertools import product
from lifelines import CoxPHFitter
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm

from machine_learning.utils.training.clean_survival_inputs import clean_survival_inputs


def _evaluate_params(df_fit, pen, l1, method, robust, step, kf_splits):
    """
    Evaluate a single hyperparameter combination across all CV folds.
    Returns (mean_c_index, params_dict) or (None, None) if all folds failed.
    """
    c_indices = []

    for train_idx, val_idx in kf_splits:
        df_train = df_fit.iloc[train_idx]
        df_val = df_fit.iloc[val_idx]

        try:
            cph = CoxPHFitter(
                penalizer=pen,
                l1_ratio=l1,
                baseline_estimation_method=method
            )
            cph.fit(
                df_train,
                duration_col="T",
                event_col="E",
                robust=robust,
                fit_options={"step_size": step}
            )
            risk_scores = cph.predict_partial_hazard(df_val)
            c_index = concordance_index_censored(
                df_val["E"].astype(bool),
                df_val["T"],
                risk_scores.values
            )[0]
            c_indices.append(c_index)
        except Exception:
            continue

    if not c_indices:
        return None, None

    return np.mean(c_indices), {
        "penalizer": pen,
        "l1_ratio": l1,
        "baseline_estimation_method": method,
        "robust": robust,
        "step_size": step
    }


def tune_cox_hyperparameters(
    X_surv, T, E,
    penalizer_grid=[0.001, 0.01, 0.1, 1, 10, 100],
    l1_ratios=[0, 0.25, 0.5, 0.75, 1],
    baseline_methods=["breslow", "efron"],
    robust_options=[True, False],
    step_sizes=[0.5, 0.75, 0.95],
    n_splits=5,
    random_state=42,
    n_jobs=-1
):
    """
    Tune CoxPHFitter hyperparameters via parallelized cross-validation.

    Parameters
    ----------
    X_surv : pd.DataFrame
        Feature matrix used for fitting the Cox model.
    T : array-like
        Time-to-event values.
    E : array-like
        Event indicator (1=event occurred, 0=censored).
    penalizer_grid : list of float
        Regularization strengths to search over.
    l1_ratios : list of float
        L1/L2 mixing ratios (0=Ridge, 1=Lasso).
    baseline_methods : list of str
        Baseline hazard estimation methods ('breslow' or 'efron').
    robust_options : list of bool
        Whether to use the robust sandwich estimator for variance.
    step_sizes : list of float
        Step sizes for the Newton-Raphson optimizer.
    n_splits : int
        Number of cross-validation folds.
    random_state : int
        Random seed for KFold splitting.
    n_jobs : int
        Number of parallel jobs for joblib (-1 uses all available cores).

    Returns
    -------
    best_params : dict
        Best hyperparameter combination found.
    best_c_index : float
        Mean cross-validated concordance index of the best params.
    """
    _, _, _, df_fit = clean_survival_inputs(X_surv, T, E)

    # Pre-compute fold splits once so all workers share the same folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kf_splits = list(kf.split(df_fit))

    # Build the full grid of hyperparameter combinations
    param_grid = list(product(
        penalizer_grid, l1_ratios, baseline_methods, robust_options, step_sizes
    ))
    total = len(param_grid)
    print(f"[tune_cox_hyperparameters] Searching {total} combinations "
          f"× {n_splits} folds using {n_jobs} workers...")

    # Run all combinations in parallel with a tqdm progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_params)(df_fit, pen, l1, method, robust, step, kf_splits)
        for pen, l1, method, robust, step in tqdm(param_grid, desc="Cox CV tuning", unit="combo")
    )

    # Pick the best result
    best_c_index, best_params = -np.inf, None
    for mean_c_index, params in results:
        if mean_c_index is not None and mean_c_index > best_c_index:
            best_c_index = mean_c_index
            best_params = params

    if best_params is None:
        print("[tune_cox_hyperparameters] No valid hyperparameters found, using defaults.")
        best_params = {
            "penalizer": 0.1,
            "l1_ratio": 0,
            "baseline_estimation_method": "breslow",
            "robust": True,
            "step_size": 0.95
        }
        best_c_index = np.nan

    print(f"[tune_cox_hyperparameters] Best Params: {best_params} | Best C-index: {best_c_index:.4f}")
    return best_params, best_c_index