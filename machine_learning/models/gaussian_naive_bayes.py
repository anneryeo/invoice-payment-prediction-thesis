import numpy as np
from sklearn.naive_bayes import GaussianNB
from .base_pipeline import BasePipeline


class GaussianNaiveBayesPipeline(BasePipeline):
    def initialize_model(self):
        """Initialize Gaussian Naive Bayes with provided parameters."""
        self.model = GaussianNB(**self.parameters)
        return self

    def fit(self, use_feature_selection=False, top_k=None):
        """Train the GaussianNB model, optionally applying custom feature selection."""
        if self.model is None:
            raise ValueError("Model not built. Call initialize_model() first.")

        self.model.fit(self.X_train, self.y_train)

        if use_feature_selection:
            means     = self.model.theta_
            variances = self.model.var_

            mean_diff        = np.abs(means.max(axis=0) - means.min(axis=0))
            influence_scores = mean_diff / variances.mean(axis=0)

            ranked_indices   = np.argsort(influence_scores)[::-1]
            selected_indices = ranked_indices[:top_k] if top_k is not None else ranked_indices

            # Build boolean mask so _set_features stays consistent with other pipelines
            mask = np.zeros(self.X_train.shape[1], dtype=bool)
            mask[selected_indices] = True

            # Capture weights before column reduction drops unselected importances
            self._set_features(
                method=f"GaussianNB influence score (top_k={top_k!r})",
                mask=mask,
                importances=influence_scores,
            )

            self.X_train = self.X_train[:, selected_indices]
            self.X_test  = self.X_test[:, selected_indices]

            self.model.fit(self.X_train, self.y_train)
        else:
            self._set_features(method="none")

        return self