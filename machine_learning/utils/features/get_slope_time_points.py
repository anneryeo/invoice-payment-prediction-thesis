import numpy as np
from lifelines import KaplanMeierFitter

def get_slope_timepoints(T, E, n_points=6, min_gap=14):
    """
    Select time points where the KM curve has steepest decline.
    min_gap prevents clustering of points too close together.
    """
    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E)
    
    sf = kmf.survival_function_["KM_estimate"].values
    timeline = kmf.survival_function_.index.values
    
    # First difference as proxy for slope magnitude
    slopes = np.abs(np.diff(sf))
    slope_times = timeline[:-1]  # times corresponding to each slope
    
    # Greedily pick top points with minimum gap between them
    sorted_idx = np.argsort(slopes)[::-1]
    selected = []
    
    for idx in sorted_idx:
        t = slope_times[idx]
        if all(abs(t - s) >= min_gap for s in selected):
            selected.append(t)
        if len(selected) == n_points:
            break
    
    raw_time_points = sorted(np.round(selected).astype(int).tolist())
    time_points = [t for t in raw_time_points if t > 0]
    
    return time_points