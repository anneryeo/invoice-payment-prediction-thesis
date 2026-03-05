import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from .base_pipeline import BasePipeline


class XGBoostPipeline(BasePipeline):
    def initialize_model(self):
        """Initialize XGBoost with provided parameters, using GPU if available."""
        try:
            import torch
            if torch.cuda.is_available():
                self.parameters.setdefault("device", "cuda")
            else:
                self.parameters.setdefault("device", "cpu")
        except ImportError:
            self.parameters.setdefault("device", "cpu")

        self.parameters.setdefault("tree_method", "hist")
        self.model = XGBClassifier(**self.parameters)
        return self

    def fit(self, use_feature_selection=False, threshold="median"):
        """Train the XGBoost model, optionally applying feature selection."""
        if self.model is None:
            raise ValueError("Model not built. Call initialize_model() first.")

        self.model.fit(self.X_train, self.y_train)

        if use_feature_selection:
            self.selector = SelectFromModel(self.model, threshold=threshold, prefit=True)

            mask = self.selector.get_support()
            self.X_train = self.selector.transform(self.X_train)
            self.X_test  = self.selector.transform(self.X_test)

            self._set_features(
                method=f"Gain-based split importance (threshold={threshold!r})",
                mask=mask,
            )

            self.model.fit(self.X_train, self.y_train)
        else:
            self._set_features(method="none")

        return self