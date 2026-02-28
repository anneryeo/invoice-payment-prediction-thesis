import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from .base_pipeline import BasePipeline

class XGBoostPipeline(BasePipeline):
    def initialize_model(self):
        """Initialize XGBoost with provided parameters, using GPU if available."""
        try:
            import xgboost
            # Simple check for GPU availability
            gpu_available = xgboost.rabit.get_rank() == 0
        except Exception:
            gpu_available = False

        if gpu_available:
            self.parameters.setdefault("tree_method", "gpu_hist")
            self.parameters.setdefault("predictor", "gpu_predictor")
        else:
            self.parameters.setdefault("tree_method", "hist")
            self.parameters.setdefault("predictor", "auto")

        self.model = XGBClassifier(**self.parameters)
        return self

    def fit(self, use_feature_selection=False, threshold="median"):
        """Train the XGBoost model, optionally applying feature selection."""
        if self.model is None:
            raise ValueError("Model not built. Call initialize_model() first.")

        # Fit model on training data
        self.model.fit(self.X_train, self.y_train)

        if use_feature_selection:
            # Fit selector
            self.selector = SelectFromModel(self.model, threshold=threshold)
            self.selector.fit(self.X_train, self.y_train)

            # Transform train/test once
            self.X_train = self.selector.transform(self.X_train)
            self.X_test  = self.selector.transform(self.X_test)

            # Save selected feature names
            if self.original_feature_names is not None:
                mask = self.selector.get_support()
                self.selected_feature_names = [
                    name for name, keep in zip(self.original_feature_names, mask) if keep
                ]

            # Retrain model on reduced features
            self.model.fit(self.X_train, self.y_train)
        else:
            # If no feature selection, keep all original names
            self.selected_feature_names = self.original_feature_names

        return self