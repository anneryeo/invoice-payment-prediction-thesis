import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize(X: pd.DataFrame) -> pd.DataFrame:
    """
    Fit StandardScaler on training data and transform the X features

    Returns
    -------
    self
    """
    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X