from sklearn.tree import DecisionTreeClassifier
from ..Utils.data_evaluation import data_evaluation

class DecisionTreePipeline:
    def __init__(self, X_train, X_test,
                 y_train, y_test,
                 args, parameters=None):
        self.args = args
        self.parameters = parameters or {}
        self.model = None
        self.results = None

        # Store pre‑prepared splits directly
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

    def build_model(self):
        # Initialize Decision Tree with provided parameters
        self.model = DecisionTreeClassifier(**self.parameters)
        return self

    def train(self):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.model.fit(self.X_train, self.y_train)
        return self

    def predict(self, X):
        """
        Generate predictions for new data.
        Accepts numpy arrays or pandas DataFrames.
        Returns numpy array of predicted class labels.
        """
        return self.model.predict(X)

    def evaluation(self):
        """
        Evaluate the model using the simplified data_evaluation
        that expects (y_pred, y_test).
        """
        y_pred = self.predict(self.X_test)
        self.results = data_evaluation(y_pred, self.y_test)
        return self

    def show_results(self):
        return self.results