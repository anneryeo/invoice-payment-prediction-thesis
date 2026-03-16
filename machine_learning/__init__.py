__all__ = [
    "AdaBoostPipeline",
    "DecisionTreePipeline",
    "GaussianNaiveBayesPipeline",
    "KNearestNeighborPipeline",
    "RandomForestPipeline",
    "XGBoostPipeline",
    "StackedEnsemblePipeline",
    "MultiLayerPerceptronPipeline",
    "TransformerPipeline",
]

import importlib

def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError("module " + __name__ + " has no attribute " + name)
    return getattr(importlib.import_module("machine_learning.models"), name)