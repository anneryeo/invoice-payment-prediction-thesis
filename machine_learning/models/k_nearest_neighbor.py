import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from .base_pipeline import BasePipeline


class KnearestNeighborPipeline(BasePipeline):
    def initialize_model(self):
        """Initialize KNN with provided parameters."""
        self.model = KNeighborsClassifier(**self.parameters)
        return self

    def fit(self, use_feature_selection=False, top_k=None, n_repeats=10, random_state=42):
        """Train the KNN model, optionally applying permutation importance feature selection."""
        if self.model is None:
            raise ValueError("Model not built. Call initialize_model() first.")

        self.model.fit(self.X_train, self.y_train)

        if use_feature_selection:
            result = permutation_importance(
                self.model, self.X_train, self.y_train,
                n_repeats=n_repeats, random_state=random_state, n_jobs=-1
            )
            importance_scores = result.importances_mean

            ranked_indices   = np.argsort(importance_scores)[::-1]
            selected_indices = ranked_indices[:top_k] if top_k is not None else ranked_indices

            # Build boolean mask so _set_features stays consistent with other pipelines
            mask = np.zeros(self.X_train.shape[1], dtype=bool)
            mask[selected_indices] = True

            self._set_features(
                method=f"Permutation Importance (top_k={top_k!r}, n_repeats={n_repeats})",
                mask=mask,
                importances=importance_scores,
            )

            self.X_train = self.X_train[:, selected_indices]
            self.X_test  = self.X_test[:, selected_indices]

            self.model.fit(self.X_train, self.y_train)
        else:
            self._set_features(method="none")

        return self