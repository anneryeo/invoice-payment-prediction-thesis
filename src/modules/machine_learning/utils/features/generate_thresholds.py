import numpy as np

def generate_thresholds(min_threshold=0.1, max_threshold=0.9, step=0.1, precision=2):
    """
    Generate thresholds incrementing by `step` between min_threshold and max_threshold.

    Parameters
    ----------
    min_threshold : float
        Starting threshold value.
    max_threshold : float
        Ending threshold value.
    step : float
        Increment step size.
    precision : int
        Decimal rounding precision.

    Returns
    -------
    list of float
    """
    num_points = int(round((max_threshold - min_threshold) / step)) + 1
    return [round(x, precision) for x in np.linspace(min_threshold, max_threshold, num_points)]
