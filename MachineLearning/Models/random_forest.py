from sklearn.ensemble import RandomForestClassifier
from ..Utils.data_evaluation import data_evaluation

class RandomForestPipeline:
    def __init__(self, X_train, X_test,
                 y_train, y_test,
                 args, parameters=None):
        self.args = args
        self.parameters = parameters
        self.model = None
        self.results = None

        # Store pre‑prepared splits directly
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

    def build_model(self):
        # Initialize Random Forest with provided parameters
        # Common parameters: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, random_state
        self.model = RandomForestClassifier(**(self.parameters or {}))
        return self

    def train(self):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.model.fit(self.X_train, self.y_train)
        return self

    def evaluation(self):
        self.results = data_evaluation(self.model, self.X_test, self.y_test)
        return self

    def show_results(self):
        return self.results