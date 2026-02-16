import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier

from .Utils.data_partitioning import data_partitioning
from .Utils.data_evaluation import data_evaluation

encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(X[['color']])
X_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['color']))

class AdaBoostPipeline:
    def __init__(self, parameters=None):
        self.parameters = parameters or {}
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.results = None

    def data_preparation(self, df_data):
        OneHotEncoder
        df_data = 

        self.X_train, self.X_test, self.y_train, self.y_test = data_partitioning(df_cs)

    def build_model(self):
        self.model = AdaBoostClassifier(**self.parameters)

    def train(self):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.model.fit(self.X_train, self.y_train)

    def feature_importance(self):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.feature_importances_

    def evaluation(self):
        self.results = data_evaluation(self.model, self.X_test, self.y_test)
        return self.results

    def show_results(self):
        print(self.results)