import numpy as np
from abc import ABC, abstractmethod
from ..Utils.data_evaluation import data_evaluation

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

        # Will be set after feature selection
        self.selected_feature_names = None

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

        # Default fit behavior (no feature selection customization)
        self.model.fit(self.X_train, self.y_train)
        self.selected_feature_names = self.original_feature_names
        return self

    def predict(self, X):
        """Generate predictions for new data."""
        X = np.array(X)
        return self.model.predict(X)

    def evaluate(self):
        """Evaluate the model using data_evaluation."""
        y_pred = self.predict(self.X_test)
        y_proba = self._predict_proba(self.X_test)
        self.results = data_evaluation(y_pred, self.y_test, y_proba=y_proba)
        return self

    def show_results(self):
        return self.results
    
    def _predict_proba(self, X):
        """Generate class probability estimates for new data."""
        X = np.array(X)
        return self.model.predict_proba(X)

    def get_selected_features(self):
        """Return the names of features selected during feature selection.
        If feature selection was not applied, return all original features."""
        if self.selected_feature_names is None:
            return self.original_feature_names
        return self.selected_feature_names