import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from .base_pipeline import BasePipeline


class XGBoostPipeline(BasePipeline):
    def initialize_model(self):
        """Initialize XGBoost with provided parameters, using GPU if available."""
        
        def xgboost_can_use_gpu():
            """Directly test if XGBoost can actually use the GPU."""
            try:
                import xgboost as xgb
                import numpy as np
                # Small test train to confirm GPU works end-to-end
                data = xgb.DMatrix(np.random.rand(10, 3), label=np.random.randint(0, 2, 10))
                params = {"device": "cuda", "tree_method": "hist", "verbosity": 0}
                xgb.train(params, data, num_boost_round=1)
                return True
            except Exception:
                return False

        if "device" not in self.parameters:
            self.parameters["device"] = "cuda" if xgboost_can_use_gpu() else "cpu"

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
                method_text=f"Gain-based split importance ",
                method_parameters=f"threshold={threshold!r}",
                mask=mask,
            )

            self.model.fit(self.X_train, self.y_train)
        else:
            self._set_features(method_text="none")

        return self