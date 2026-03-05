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

        # Fit model on training data
        self.model.fit(self.X_train, self.y_train)

        if use_feature_selection:
            # GaussianNB exposes theta_ (means) and var_ (variances)
            means = self.model.theta_      # shape: (n_classes, n_features)
            variances = self.model.var_    # shape: (n_classes, n_features)

            # Influence score: mean difference across classes / average variance
            mean_diff = np.abs(means.max(axis=0) - means.min(axis=0))
            influence_scores = mean_diff / variances.mean(axis=0)

            ranked_indices = np.argsort(influence_scores)[::-1]

            if top_k is None:
                selected_indices = ranked_indices
            else:
                selected_indices = ranked_indices[:top_k]

            # Reduce train/test sets
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