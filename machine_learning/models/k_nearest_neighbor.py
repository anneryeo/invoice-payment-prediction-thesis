import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from .base_pipeline import BasePipeline

class KnearestNeighborPipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_importances = None

    def initialize_model(self):
        """Initialize KNN with provided parameters."""
        self.model = KNeighborsClassifier(**self.parameters)
        return self

    def fit(self, use_feature_selection=False, top_k=None, n_repeats=10, random_state=42):
        """Train the KNN model, optionally applying permutation importance feature selection."""
        if self.model is None:
            raise ValueError("Model not built. Call initialize_model() first.")

        # Fit model on training data
        self.model.fit(self.X_train, self.y_train)

        if use_feature_selection:
            # Compute permutation importance
            result = permutation_importance(
                self.model, self.X_train, self.y_train,
                n_repeats=n_repeats, random_state=random_state, n_jobs=-1
            )
            self.feature_importances = result.importances_mean

            # Rank features by importance
            ranked_indices = np.argsort(self.feature_importances)[::-1]

            # Select top_k features (or all if None)
            if top_k is None:
                selected_indices = ranked_indices
            else:
                selected_indices = ranked_indices[:top_k]

            # Apply selection
            self.X_train = self.X_train[:, selected_indices]
            self.X_test  = self.X_test[:, selected_indices]

            # Save selected feature names
            if self.original_feature_names is not None:
                self.selected_feature_names = [
                    self.original_feature_names[i] for i in selected_indices
                ]

            # Retrain model on reduced features
            self.model.fit(self.X_train, self.y_train)
        else:
            # If no feature selection, keep all original names
            self.selected_feature_names = self.original_feature_names

        return self

    def get_feature_importances(self):
        """Return permutation importance scores for all features.
        If feature selection was not applied, return None."""
        return self.feature_importances