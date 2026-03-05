import numpy as np    

def adjust_payment_period(T, ε=1e-6):
    """
    Adjusts the payment period to avoid negative or zero values.

    Parameters
    ----------
    T : array-like or float
        The time period(s) to adjust.
    epsilon : float, optional
        A small constant added to avoid zero values (default is 1e-6).

    Returns
    -------
    adjusted_T : ndarray or float
        The adjusted time period(s).
    earliest_pre_payment : ndarray or float
        The portion representing pre-payments (non-positive values).
    """
    earliest_pre_payment = np.minimum(T, 0)
    adjusted_T = T - earliest_pre_payment + ε
    return adjusted_T