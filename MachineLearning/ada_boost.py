import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

from .Utils.data_partitioning import data_partitioning
from .Utils.data_evaluation import data_evaluation

class AdaBoostPipeline:
    def __init__(self, parameters=None):
        self.parameters = parameters or {}
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.results = None

    def data_preparation(self, df_data):
        self.X_train, self.X_test, self.y_train, self.y_test = data_partitioning(df_data)
        return self

    def build_model(self):
        self.model = AdaBoostClassifier(**self.parameters)
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
        print(self.results)
        return self