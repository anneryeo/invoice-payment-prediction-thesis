# machine_learning/utils/io/analysis/__init__.py
#
# Public API for the fluent results analysis module.
#
# Usage:
#     from analysis import ResultsAnalyzer
#     ra = ResultsAnalyzer("data/training_results/")
#     ra.top(10).charts("confusion_matrix").plot()

from .analyzer import ResultsAnalyzer
from .quality import QualityReport
from .registry import (
    ALL_MODELS,
    BASE_MODELS,
    CLASS_NAMES,
    FAMILY_MAP,
    FAMILY_PALETTE,
    MODEL_DISPLAY,
    ORDINAL_BASE_MAP,
    ORDINAL_MODELS,
    STRATEGY_LABELS,
    STRATEGY_ORDER,
    SURVIVAL_FEATURES,
    TWO_STAGE_BASE_MAP,
    TWO_STAGE_MODELS,
)
from .result_set import (
    ChartCollection,
    ChartItem,
    ComparisonResult,
    FeatureResult,
    ResultSet,
)
from .visualization import Theme

__all__ = [
    "ResultsAnalyzer",
    "ResultSet",
    "ChartCollection",
    "ChartItem",
    "ComparisonResult",
    "FeatureResult",
    "QualityReport",
    "Theme",
    # Registry re-exports
    "ALL_MODELS",
    "BASE_MODELS",
    "ORDINAL_MODELS",
    "TWO_STAGE_MODELS",
    "MODEL_DISPLAY",
    "FAMILY_MAP",
    "FAMILY_PALETTE",
    "STRATEGY_ORDER",
    "STRATEGY_LABELS",
    "CLASS_NAMES",
    "ORDINAL_BASE_MAP",
    "TWO_STAGE_BASE_MAP",
    "SURVIVAL_FEATURES",
]
