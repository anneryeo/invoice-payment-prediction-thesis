__all__ = [
    "AdaBoostPipeline",
    "DecisionTreePipeline",
    "GaussianNaiveBayesPipeline",
    "KnearestNeighborPipeline",
    "RandomForestPipeline",
    "XGBoostPipeline",
    "MultiLayerPerceptronPipeline",
    "TransformerPipeline",
]

# Lazy imports: only load when accessed
import importlib

def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    module_map = {
        "AdaBoostPipeline": "machine_learning.models.ada_boost",
        "DecisionTreePipeline": "machine_learning.models.decision_tree",
        "GaussianNaiveBayesPipeline": "machine_learning.models.gaussian_naive_bayes",
        "KnearestNeighborPipeline": "machine_learning.models.k_nearest_neighbor",
        "RandomForestPipeline": "machine_learning.models.random_forest",
        "XGBoostPipeline": "machine_learning.models.xg_boost",
        "MultiLayerPerceptronPipeline": "machine_learning.models.multi_layer_perceptron",
        "TransformerPipeline": "machine_learning.models.transformer",
    }

    module = importlib.import_module(module_map[name])
    return getattr(module, name)