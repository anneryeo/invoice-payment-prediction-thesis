import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from ..utils.training.data_evaluation import data_evaluation


@dataclass
class FeatureInfo:
    method: Optional[str] = None
    selected: Optional[list] = None
    weights: Optional[dict] = None


class BasePipeline(ABC):
    def __init__(self, X_train, X_test,
                 y_train, y_test,
                 args, parameters=None,
                 feature_names=None):
        self.args = args
        self.parameters = parameters or {}
        self.model = None
        self.selector = None
        self.results = None

        # Store original feature names if provided (e.g. from DataFrame)
        if feature_names is None and hasattr(X_train, "columns"):
            feature_names = list(X_train.columns)
        self.original_feature_names = feature_names

        # Convert everything to NumPy arrays (default handling)
        self.X_train = np.array(X_train)
        self.X_test  = np.array(X_test)
        self.y_train = np.array(y_train)
        self.y_test  = np.array(y_test)

        # Initialise feature info — populated after fit()
        self.features = FeatureInfo()

    @abstractmethod
    def initialize_model(self):
        """Initialize the model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def fit(self, use_feature_selection=False, **kwargs):
        """
        Train the model on the training set.
        Subclasses override this to implement their own feature selection logic.
        """
        if self.model is None:
            raise ValueError("Model not built. Call initialize_model() first.")

        self.model.fit(self.X_train, self.y_train)
        self._set_features(method="none")
        return self

    def _set_features(self, method, mask=None, importances=None):
        """
        Populate self.features (a FeatureInfo dataclass) after a model has been fitted.

        Should be called at the end of fit() in every subclass — once when no feature
        selection is used (mask=None), or once before retraining when a selection mask
        is available.

        Parameters
        ----------
        method : str
            A human-readable description of the feature selection strategy applied.
            Use "none" when no selection was performed, or a descriptive string such
            as "SelectFromModel(threshold='median')" when a selector was used.

        mask : array-like of bool, optional
            A boolean array of shape (n_original_features,) where True indicates a
            feature was selected. If None, all original features are assumed to have
            been kept. Typically obtained via self.selector.get_support().

        importances : array-like of float, optional
            Importance scores for all original features, aligned by index. If None,
            falls back to the model's feature_importances_ attribute if available.
            Explicit values are required for models that use custom scoring (e.g.
            GaussianNB influence scores, KNN permutation importance).

        Sets
        ----
        self.features.method : str
            Mirrors the method argument directly.

        self.features.selected : list of str or None
            Names of the features that were kept. None if original_feature_names
            was not provided at construction time.

        self.features.weights : dict of {str: float} or None
            Feature importance scores (rounded to 6 decimal places) for selected
            features only, keyed by feature name. Sourced from the model's
            feature_importances_ attribute. None if original_feature_names was not
            provided, or if the model does not expose feature_importances_.

        Notes
        -----
        Weights are captured from the model's feature_importances_ at the time of
        the call. When feature selection is used, this should be called before
        retraining on the reduced feature set — otherwise the importances will
        reflect the retrained model and may no longer align with the original
        feature indices.
        """
        names = self.original_feature_names

        if importances is None and hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_

        if mask is None:
            selected = names
            weights  = (
                {n: round(float(s), 6) for n, s in zip(names, importances)}
                if names is not None and importances is not None else None
            )
        else:
            selected = (
                [n for n, keep in zip(names, mask) if keep]
                if names is not None else None
            )
            weights = (
                {n: round(float(s), 6)
                for n, s, keep in zip(names, importances, mask) if keep}
                if names is not None and importances is not None else None
            )

        self.features = FeatureInfo(method=method, selected=selected, weights=weights)

    def predict(self, X):
        """Generate predictions for new data."""
        return self.model.predict(np.array(X))

    def evaluate(self):
        """Evaluate the model using data_evaluation."""
        y_pred  = self.predict(self.X_test)
        y_proba = self._predict_proba(self.X_test)
        self.results = data_evaluation(y_pred, self.y_test, y_proba=y_proba)
        return self

    def show_results(self):
        return self.results

    def _predict_proba(self, X):
        """Generate class probability estimates for new data."""
        return self.model.predict_proba(np.array(X))

    def get_selected_features(self):
        """Return selected feature names, falling back to all originals."""
        return self.features.selected or self.original_feature_names