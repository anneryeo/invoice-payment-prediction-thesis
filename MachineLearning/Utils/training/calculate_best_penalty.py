import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import warnings

# Suppress scikit-learn UserWarning: "all coefficients are zero, consider decreasing alpha"
# This warning occurs when the chosen alpha value is too high, causing the model to shrink
# all coefficients to zero. We silence it here because this behavior is expected in some runs.
warnings.filterwarnings("ignore", message=".*all coefficients are zero.*")
warnings.filterwarnings("ignore", message=".*all coefficients are zero.*")

def calculate_best_penalty(df_data_surv):
    # ============================================
    # STEP 1 — Initialize and Prepare Data
    # ============================================

    print("extracing variables")
    X = df_data_surv.drop(columns=['days_elapsed_until_fully_paid', 'censor'])
    T = df_data_surv['days_elapsed_until_fully_paid']
    E = df_data_surv['censor']

    # Avoid negative values by shifting by the days of pre-paid period
    earliest_pre_payment = np.minimum(T, 0) # Maximum to only get pre-payments
    ε = 1e-6 # Used to avoid zero values
    T = T - earliest_pre_payment + ε

    # Build survival array
    survival_train = np.array(
        [(bool(e), float(t)) for e, t in zip(E.values, T.values)],
        dtype=[('event', 'bool'), ('time', 'float')]
    )

    # ============================================
    # STEP 2 — Tune Survival Model
    # ============================================
    penalties = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    best_penalty, best_c_index = None, -np.inf

    for λ in penalties:
        model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[λ], normalize=True)
        model.fit(X, survival_train)
        risk_scores = model.predict(X)
        c_index = concordance_index_censored(E.astype(bool), T, risk_scores)[0]
        if c_index > best_c_index:
            best_penalty, best_c_index = λ, c_index

    print(f"Best Penalty: {best_penalty} | Best C-index: {best_c_index}")

    return best_penalty