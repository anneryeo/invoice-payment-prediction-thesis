__all__ = [
    "AdaBoostPipeline",
    "DecisionTreePipeline",
    "GaussianNaiveBayesPipeline",
    "KNearestNeighborPipeline",
    "RandomForestPipeline",
    "XGBoostPipeline",
    "StackedEnsemblePipeline",
    "OrdinalPipeline",
    "TwoStagePipeline",
]

# Lazy imports: only load when accessed
import importlib

def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError("module " + __name__ + " has no attribute " + name)
    
    module_map = {
        "AdaBoostPipeline": "machine_learning.models.ada_boost",
        "DecisionTreePipeline": "machine_learning.models.decision_tree",
        "GaussianNaiveBayesPipeline": "machine_learning.models.gaussian_naive_bayes",
        "KNearestNeighborPipeline": "machine_learning.models.k_nearest_neighbor",
        "RandomForestPipeline": "machine_learning.models.random_forest",
        "XGBoostPipeline": "machine_learning.models.xg_boost",
        "StackedEnsemblePipeline": "machine_learning.models.stacked_ensemble",
        "OrdinalPipeline": "machine_learning.models.ordinal_classifier",
        "TwoStagePipeline": "machine_learning.models.two_stage_classifier",
    }

    module = importlib.import_module(module_map[name])
    return getattr(module, name)