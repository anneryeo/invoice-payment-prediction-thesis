import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def data_evaluation(model, X_test, y_test):
    """
    Evaluate a trained model on test data and return results as a dictionary.
    """
    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    return results