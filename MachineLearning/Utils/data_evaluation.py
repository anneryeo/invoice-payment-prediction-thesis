from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def data_evaluation(y_pred, y_test, y_proba=None):
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }
    
    # Add AUC only if probability scores are provided
    if y_proba is not None:
        try:
            results["roc_auc_macro"] = roc_auc_score(y_test, y_proba, average="macro", multi_class="ovr")
        except ValueError:
            results["roc_auc_macro"] = None  # Handle cases where AUC cannot be computed
    
    return results