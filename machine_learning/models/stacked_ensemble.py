from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import numpy as np
from .base_pipeline import BasePipeline


class StackedEnsemblePipeline(BasePipeline):
    BASE_MODELS = {
        "adaboost": AdaBoostClassifier,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "bagging": BaggingClassifier,
    }

    def initialize_model(self, estimators=None, final_estimator=None):
        """
        Initialize a stacking ensemble with chosen base models.

        Parameters
        ----------
        estimators : dict of {str: dict} or None
            Maps each base model key to its own parameter dict.
            If None, defaults to adaboost and random_forest with self.parameters.

            Example
            -------
            estimators={
                "adaboost":          {"learning_rate": 0.1, "n_estimators": 10},
                "random_forest":     {"max_depth": 10, "min_samples_leaf": 1, "n_estimators": 100},
                "gradient_boosting": {"learning_rate": 0.05, "n_estimators": 50},
                "bagging":           {"n_estimators": 20},
            }

        final_estimator : sklearn estimator
            Meta-model to combine base learners (default LogisticRegression).
        """
        if estimators is None:
            estimators = {
                "adaboost":      self.parameters,
                "random_forest": self.parameters,
            }

        stacking_estimators = []
        for name, params in estimators.items():
            if name not in self.BASE_MODELS:
                raise ValueError(f"Unsupported ensemble type: {name}")
            model_cls = self.BASE_MODELS[name]
            stacking_estimators.append((name, model_cls(**params)))

        if final_estimator is None:
            final_estimator = LogisticRegression()

        self.model = StackingClassifier(
            estimators=stacking_estimators,
            final_estimator=final_estimator,
            passthrough=False
        )
        self.model_types = list(estimators.keys())
        return self

    def fit(self, use_feature_selection=False, threshold="median",
            n_repeats=5, random_state=None):
        """
        Train the stacked ensemble, optionally applying feature selection.

        When use_feature_selection=True, permutation importance is computed on
        the full StackingClassifier — the only model-agnostic importance measure
        that reflects the ensemble's combined learned behaviour. Features whose
        mean permutation importance falls below the threshold are dropped, then
        the ensemble is retrained on the reduced feature set.

        Parameters
        ----------
        use_feature_selection : bool
            Whether to apply feature selection before final training.
        threshold : float or "median" or "mean"
            Features whose mean permutation importance is below this value are
            dropped. Mirrors the SelectFromModel threshold semantics:
              - "median" : keep features above the median importance
              - "mean"   : keep features above the mean importance
              - float    : keep features with importance >= that value
        n_repeats : int
            Number of times to permute each feature. Higher values yield more
            stable importance estimates at the cost of runtime (default 5).
        random_state : int or None
            Seed for reproducibility of the permutation shuffles.
        """
        if self.model is None:
            raise ValueError("Model not built. Call initialize_model() first.")

        self.model.fit(self.X_train, self.y_train)

        if use_feature_selection:
            # StackingClassifier exposes neither coef_ nor feature_importances_,
            # so SelectFromModel cannot be used directly. Permutation importance
            # is the correct alternative — it measures how much each feature
            # affects *this* ensemble's predictions by observing score drops
            # when each feature is shuffled, making it fully model-agnostic.
            perm = permutation_importance(
                self.model, self.X_train, self.y_train,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=-1,
            )
            importances = perm.importances_mean  # shape (n_features,)

            # Resolve threshold string to a scalar cutoff
            if threshold == "median":
                cutoff = np.median(importances)
            elif threshold == "mean":
                cutoff = np.mean(importances)
            else:
                cutoff = float(threshold)

            mask = importances >= cutoff

            if not mask.any():
                raise ValueError(
                    "Permutation importance threshold eliminated all features. "
                    "Lower the threshold or disable feature selection."
                )

            self.X_train = self.X_train[:, mask]
            self.X_test  = self.X_test[:, mask]

            self._set_features(
                method_text="Permutation importance",
                method_parameters=f"threshold={threshold!r}, n_repeats={n_repeats}",
                mask=mask,
                importances=importances,
            )

            # Retrain the full stacking ensemble on the reduced feature set
            self.model.fit(self.X_train, self.y_train)
        else:
            self._set_features(method_text="none", method_parameters="none")

        return self