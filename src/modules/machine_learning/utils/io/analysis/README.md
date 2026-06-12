# ResultsAnalyzer — Fluent API for ML Experiment Analysis

A composable, chainable Python API for exploring invoice payment prediction
experiment results stored in SQLite. Replaces the 1,700-line procedural
`ML_Results_Analysis.py` notebook with a reusable module that supports lazy
loading, immutable query chains, and publication-quality visualizations.

## Installation

The module lives inside the existing `io/` package. No additional
dependencies beyond what the project already uses (pandas, numpy,
matplotlib, seaborn, scipy).

Drop the `analysis/` folder into `src/modules/machine_learning/utils/io/`
and replace `__init__.py` with the updated version from the zip.

## Quick start

```python
from machine_learning.utils.io import ResultsAnalyzer

ra = ResultsAnalyzer("data/training_results/")

# Top 10 experiments by F1
ra.top(10)

# Best per model family, show confusion matrices
ra.best_per("family").top(3).charts("confusion_matrix").plot()

# Heatmap of F1 across all models and strategies
ra.all().plot(kind="heatmap")

# Statistical comparison
ts = ra.family("Two-Stage").top(1)
base = ra.family("Base").top(1)
print(ts.compare(base, test="mannwhitney"))
```

## Architecture

### Design pattern — composition over inheritance

`ResultsAnalyzer` wraps `SessionStore` and `ResultsRepository` via
composition, not inheritance. Both composed objects are **public
attributes** so existing code (Dash dashboard, training pipeline) can
access them directly as escape hatches.

```
ResultsAnalyzer
├── .store  →  SessionStore       (filesystem navigation)
├── .repo   →  ResultsRepository  (SQLite I/O, current session)
└── .df     →  pd.DataFrame       (lazy-loaded, cached, enriched)
```

Rationale:

- `SessionStore` is a filesystem navigator; `ResultsAnalyzer` is an
  analysis engine — "uses" not "is-a"
- The Dash dashboard already imports `SessionStore` directly; inheritance
  could break that contract
- `use_session()` tears down cached state — cleaner with composition
- Testable: inject a mock via `ResultsAnalyzer.from_repository(mock_repo)`

### File structure

```
io/
├── __init__.py                        # Updated: groups exports by concern
├── db_schema.py                       # UNCHANGED — DDL strings
├── results_repository.py              # UNCHANGED — OOP SQLite interface
├── load_results_from_folder.py        # UNCHANGED — SessionStore
├── save_results_to_folder.py          # UNCHANGED — writer
├── data_loaders.py                    # UNCHANGED — Dash helpers
├── migrate_db_schema.py               # UNCHANGED — schema migrators
└── analysis/                          # NEW
    ├── __init__.py                    # Public API re-exports
    ├── analyzer.py                    # ResultsAnalyzer entry point
    ├── result_set.py                  # ResultSet, ChartCollection, etc.
    ├── registry.py                    # Model/strategy metadata constants
    ├── quality.py                     # QualityReport
    └── visualization/
        ├── __init__.py                # Re-exports Theme
        ├── theme.py                   # Design tokens, rcParams presets
        └── plots.py                   # All plot rendering functions
```

### Core abstraction — immutable ResultSet

Every filtering/ranking method returns a **new** `ResultSet`. The original
is never mutated. This makes chains safe and predictable.

```python
# These are independent — filtering one doesn't affect the other
all_base = ra.family("Base")
top_base = all_base.top(5)
best_base = all_base.best_per("model")
```

### Lazy loading strategy

| Access                        | What loads                              | When       |
|-------------------------------|-----------------------------------------|------------|
| `ra.df`                       | experiments + metrics tables (JOIN)      | First access (~2ms) |
| `.charts("confusion_matrix")` | charts table, scoped to selection        | On iteration/access |
| `.features()`                 | features table, scoped to selection      | On `.as_dict()` or `.plot()` |
| `ra.metadata`                 | metadata blob table                     | First access |
| `ra.class_mappings`           | class_mappings blob table               | First access |
| `ra.survival`                 | survival_results blob table             | First access |

## API reference

### ResultsAnalyzer — entry point

#### Constructors

```python
# Standard — bind to most recent session
ra = ResultsAnalyzer("data/training_results/")

# Specific session by name or index
ra = ResultsAnalyzer("data/training_results/", session="2026_04_25_01")
ra = ResultsAnalyzer("data/training_results/", session=0)  # 0 = newest

# From a specific database file
ra = ResultsAnalyzer.from_db("path/to/results.db")

# From an existing ResultsRepository (useful for testing)
ra = ResultsAnalyzer.from_repository(mock_repo)
```

#### Properties

| Property          | Type                | Description                              |
|-------------------|---------------------|------------------------------------------|
| `ra.store`        | `SessionStore`      | Filesystem navigator (public)            |
| `ra.repo`         | `ResultsRepository` | SQLite interface for current session     |
| `ra.df`           | `pd.DataFrame`      | Enriched experiment DataFrame (cached)   |
| `ra.metadata`     | `dict`              | Session metadata (timestamps, models)    |
| `ra.class_mappings` | `dict`            | Label → integer encoding                 |
| `ra.survival`     | `dict`              | Survival analysis results                |
| `ra.sessions`     | `list[str]`         | Available session folder names           |
| `ra.current_session` | `str`            | Currently bound session                  |

#### Session management

```python
ra.use_session(0)                    # Switch by index (0 = newest)
ra.use_session("2026_04_25_01")      # Switch by name
ra.use_session()                     # Reset to most recent
```

Clears all cached data. Returns `self` for chaining.

#### Summary and quality

```python
ra.summary()    # Prints formatted session summary, returns string
ra.quality()    # Returns QualityReport dataclass
```

#### Query entry points

All return a `ResultSet`:

```python
ra.all()                              # Everything
ra.top(10)                            # Top N by enhanced_f1_macro
ra.top(5, by="enhanced_roc_auc_macro") # Top N by any metric
ra.best_per("model")                  # Best experiment per model
ra.best_per("family")                 # Best experiment per family
ra.family("Base")                     # Filter by family
ra.family("Ordinal", "Two-Stage")     # Multiple families
ra.model("xgboost")                   # Filter by model slug
ra.strategy("smote", "none")          # Filter by balance strategy
ra.where(enhanced_f1_macro__gt=0.55)  # Django-style lookup
ra.where(lambda df: df["f1_lift"] > 0.05)  # Predicate function
```

### ResultSet — chainable query object

Every method returns a new `ResultSet`:

#### Filtering

```python
rs.family("Base", "Ordinal")
rs.model("xgboost", "random_forest")
rs.strategy("smote", "hybrid@0.7")
rs.where(enhanced_f1_macro__gt=0.5)
rs.where(lambda df: df["f1_lift"] > 0.05)
```

Supported `where()` suffixes: `__gt`, `__lt`, `__gte`, `__lte`, `__eq`,
`__ne`, `__in`.

#### Ranking

```python
rs.top(10)                             # Top N by F1 (default)
rs.top(5, by="enhanced_roc_auc_macro") # Top N by any column
rs.best_per("model")                   # Best per group
rs.best_per("strategy_label")
```

#### Aggregation

```python
rs.aggregate("max")                    # Max metrics per model×strategy
rs.aggregate("mean", group_by=("family",))
rs.pivot(index="model", columns="strategy_label",
         values="enhanced_f1_macro")   # Returns raw DataFrame
```

#### Variant analysis

```python
rs.variant_lift()  # DataFrame comparing ordinal/two-stage vs base models
```

#### Terminal operations

```python
# Charts — lazy, fetches from DB only when accessed
charts = rs.charts("confusion_matrix")          # ChartCollection
charts = rs.charts("roc_curve", phase="enhanced")
charts.plot(save="figures/cm.png")

# Features
feat = rs.features()                            # FeatureResult
feat.top(20).plot(highlight_survival=True)
feat.survival_share()                           # dict[model → float]

# Statistical comparison
result = rs.compare(other_rs, test="mannwhitney")
result = rs.compare(other_rs, test="spearman")
result = rs.rank_correlation("enhanced_f1_macro", "enhanced_roc_auc_macro")

# Visualization
rs.plot()                                       # Auto-dispatched
rs.plot(kind="bar")                             # Horizontal bar chart
rs.plot(kind="heatmap")                         # Model × strategy heatmap
rs.plot(kind="grouped_bar", metric="enhanced_f1_macro")
rs.plot(save="figures/output.png", figsize=(14, 6))
```

### ChartCollection

Returned by `ResultSet.charts()`. Lazy — fetches from DB on first access.

```python
cms = ra.top(3).charts("confusion_matrix")

len(cms)           # Number of charts
cms[0].data        # Raw matrix data (list of lists)
cms[0].model       # Model slug
cms[0].model_display  # Human-readable name
cms[0].strategy    # Balance strategy
cms[0].f1          # F1 score
cms[0].auc         # AUC score

for chart in cms:
    print(chart)

cms.plot(save="figures/cm_top3.png")
```

### FeatureResult

Returned by `ResultSet.features()`.

```python
feat = ra.model("xgboost").top(1).features()

feat.as_dict()          # {model_slug: [(feature, weight), ...]}
feat.top(20)            # New FeatureResult limited to top 20
feat.survival_share()   # {model_slug: fraction_survival_features}
feat.plot(highlight_survival=True)
```

### ComparisonResult

Returned by `ResultSet.compare()` and `ResultSet.rank_correlation()`.

```python
result = ts.compare(base, test="mannwhitney")

result.label_a       # Description of group A
result.label_b       # Description of group B
result.test          # Test name
result.statistic     # Test statistic value
result.p_value       # p-value
result.significant   # True if p < 0.05

print(result)        # Formatted output
```

### QualityReport

Returned by `ResultsAnalyzer.quality()`.

```python
qr = ra.quality()

qr.passed           # bool
qr.issues           # list[str]
qr.null_counts      # dict[column → count]
qr.strategy_counts  # pd.Series
qr.model_counts     # pd.Series
qr.hybrid_nulls     # int
qr.mean_lift        # float
qr.negative_lifts   # int

print(qr)            # Formatted report with ✓/⚠ status
```

### Theme

```python
from machine_learning.utils.io.analysis.visualization import Theme

Theme.apply()              # Notebook preset (serif, 120 DPI)
Theme.apply("publication") # Publication preset (sans-serif, styled bg)
Theme.reset()              # Restore matplotlib defaults
```

## Registry constants

All model/strategy metadata lives in `analysis/registry.py`:

```python
from machine_learning.utils.io.analysis.registry import (
    ALL_MODELS,            # All 15 model slugs
    BASE_MODELS,           # 6 base classifiers
    ORDINAL_MODELS,        # 3 ordinal models
    TWO_STAGE_MODELS,      # 6 two-stage models
    MODEL_DISPLAY,         # slug → human name
    FAMILY_MAP,            # slug → "Base"/"Ordinal"/"Two-Stage"
    ORDINAL_BASE_MAP,      # ordinal slug → base slug
    TWO_STAGE_BASE_MAP,    # two-stage slug → base slug
    STRATEGY_ORDER,        # canonical strategy ordering
    STRATEGY_LABELS,       # slug → human label
    CLASS_NAMES,           # payment class labels
    SURVIVAL_FEATURES,     # set of survival-derived feature names
    FAMILY_PALETTE,        # family → hex color
    STRATEGY_PALETTE,      # strategy → hex color
)

# Helper functions
from machine_learning.utils.io.analysis.registry import (
    display_name,          # model slug → display name
    family,                # model slug → family name
    strategy_label,        # (strategy, threshold) → label
)
```

## Full workflow — replacing the notebook

The entire 1,700-line notebook reduces to roughly 30 lines:

```python
ra = ResultsAnalyzer("data/training_results/")
Theme.apply()

# Load + quality (Sections 1-2)
ra.summary()
assert ra.quality().passed

# Rankings (Section 3)
ra.top(20).plot(save="figures/top20_f1.png")
ra.top(20, by="enhanced_roc_auc_macro").plot(save="figures/top20_auc.png")
print(ra.rank_correlation())

# Best per model (Section 4)
ra.best_per("model").plot(save="figures/best_per_model.png")

# Variant lift (Section 5)
print(ra.variant_lift())

# Strategy analysis (Sections 6-7)
for metric in ["enhanced_f1_macro", "enhanced_roc_auc_macro"]:
    ra.all().plot(kind="grouped_bar", metric=metric,
                  save=f"figures/bar_{metric}.png")
    ra.all().plot(kind="heatmap", metric=metric,
                  save=f"figures/heatmap_{metric}.png")

# Charts (Sections 8-9)
ra.best_per("model").top(3).charts("confusion_matrix").plot(
    save="figures/cm.png")
ra.top(5, by="enhanced_roc_auc_macro").charts("roc_curve").plot(
    save="figures/roc.png")

# Feature importance (Section 10)
for m in ["xgboost", "random_forest"]:
    ra.model(m).top(1).features().top(20).plot(
        save=f"figures/fi_{m}.png", highlight_survival=True)

# Statistical tests (Section 11)
print(ra.family("Two-Stage").top(1).compare(
    ra.family("Base").top(1), test="mannwhitney"))
```

## Compatibility

- All existing files in `io/` are unchanged
- The Dash dashboard continues to import `SessionStore` and
  `ResultsRepository` directly — no modifications needed
- The training pipeline continues to use `save_training_results()` as before
- `ResultsAnalyzer.store` and `ResultsAnalyzer.repo` provide direct access
  to the wrapped objects for any code that needs them

## Known limitations

- McNemar's test requires re-running the ML pipeline to get per-sample
  predictions; the fluent API supports `spearman` and `mannwhitney` tests
  on aggregate metrics but not `mcnemar`
- PR curve plotting is not yet implemented in `plots.py` (ROC and confusion
  matrix are supported)
- The `compare()` method operates on aggregate metric distributions, not
  per-sample predictions
