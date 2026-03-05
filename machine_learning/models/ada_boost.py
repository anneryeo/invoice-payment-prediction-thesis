from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from .base_pipeline import BasePipeline


class AdaBoostPipeline(BasePipeline):
    def initialize_model(self):
        """Initialize AdaBoost with provided parameters."""
        self.model = AdaBoostClassifier(**self.parameters)
        return self

    def fit(self, use_feature_selection=False, threshold="median"):
        """Train the AdaBoost model, optionally applying feature selection."""
        if self.model is None:
            raise ValueError("Model not built. Call initialize_model() first.")

        self.model.fit(self.X_train, self.y_train)

        if use_feature_selection:
            self.selector = SelectFromModel(self.model, threshold=threshold, prefit=True)

            mask = self.selector.get_support()
            self.X_train = self.selector.transform(self.X_train)
            self.X_test  = self.selector.transform(self.X_test)

            # Capture weights before retrain overwrites importances
            self._set_features(
                method=f"Weighted estimator votes, threshold={threshold!r})",
                mask=mask
            )

            self.model.fit(self.X_train, self.y_train)
        else:
            self._set_features(method="none")

        return self