# Project Plan: IPPP Machine Learning Optimization

## Phase 4: Pipeline Performance & Caching Strategy

The current experiment pipeline executes 1092 tasks and takes approximately 3 hours. Analysis identifies significant redundant computation in feature generation (Cox PH survival features and LDA transformations) which are currently re-computed for every model/parameter combination within a balancing strategy.

### Goal
Reduce total execution time by 50-70% by implementing a multi-layer caching strategy and optimizing redundant data transformations.

### 1. Identify Redundant Computations
- **Survival Feature Generation:** Currently computed once per (strategy, threshold) pair, but passed as a full dataset to workers.
- **LDA Transformation:** Computed once per (strategy, threshold) pair.
- **Data Resampling:** SMOTE and variants are currently re-run for each (strategy, threshold) pair.

### 2. Implementation Strategy

#### Layer 1: Disk-Based Dataset Caching
Modify `SurvivalExperimentRunner` to cache the fully prepared dataset (resampled + survival features + LDA) for each `(balance_strategy, threshold)` pair.
- **Location:** `data/cache/datasets/`
- **Key:** `hash(balance_strategy, threshold, observation_end, test_size)`
- **Tool:** `joblib.dump` for efficient storage of large DataFrames.

#### Layer 2: Model Component Caching
Cache the fitted `StandardScaler`, `CoxnetSurvivalAnalysis`, and `LDATransformer` instances to allow rapid re-generation if underlying data changes slightly.
- **Location:** `data/cache/models/`

#### Layer 3: Task-Level Checkpointing (Existing)
Maintain and improve the existing `_checkpoint.pkl` to ensure zero re-run of successful model fits if a 3-hour run is interrupted.

### 3. Execution Steps
1.  **Create Cache Utility:** Implement `src/modules/machine_learning/utils/training/cache_manager.py`.
2.  **Refactor `prepare_dataset`:** Update `SurvivalExperimentRunner.prepare_dataset` to check cache before computing.
3.  **Parallel Pre-computation:** Before starting the model Pool, pre-compute and cache all strategy-specific datasets sequentially (to avoid redundant disk writes).
4.  **Worker Optimization:** Pass only the *path* to the cached dataset to workers, or rely on shared memory if possible, to reduce pickling overhead.

### 4. Validation
- Verify that cached results are identical to fresh-computed results.
- Benchmark a small subset (e.g., 2 models, 2 strategies) to measure time savings.
- Run a full 1092-experiment sweep and document the new total runtime.
