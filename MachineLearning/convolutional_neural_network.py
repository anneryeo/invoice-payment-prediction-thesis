import pandas as pd
from Utils.data_partitioning import data_partitioning
from MachineLearning.Utils.data_evaluation import data_validation
from Utils.load_parameters import load_parameters

class AdaBoostPipeline():
    def __init__(self, df_cs):
        self._data_preparation(df_cs)
        self._feature_importance()
        self._train()
        self._evaluation()

    def _data_preparation(self, df):
        data_partitioning(df)

        return df

    def _feature_importance(self):
        pass

    def _train(self):
        pass

    def _evaluation(self):
        pass

    def show_results(self):
        pass