# Code Pipeline Trace

_Source of truth: step_3.py call graph and module exploration._

Steps marked **[NEW]** are in the code but absent from the current Level-1 DFDs.

## Step 1.0: Data Importation

**Source**: `src/app/screens/initial_setup/callbacks/step_3.py → clean_datasets()`  
**Function**: `clean_datasets(revenues_content, enrollees_content)`

**Inputs**:
- Revenue Ledger (.xlsx) — base64-encoded
- Enrollee Information (.xlsx) — base64-encoded

**Outputs**:
- Raw revenue DataFrame
- Raw enrollee DataFrame

**Notes**: Base64 decoding + Excel read via calamine engine. Async parallel file loads in Revenues loader.

---

## Step 2.0: Time to Payment (DTP)

**Source**: `src/modules/feature_engineering/credit_sales_machine_learning.py → FeatureEngineer._merge_dtp()`  
**Function**: `_merge_dtp(df)`

**Inputs**:
- Matched payment records
- Due dates per receivable

**Outputs**:
- DTP lag features: dtp_1, dtp_2, dtp_3, dtp_4 (days from due to fully paid, per prior invoice)

**Notes**: Vectorized pandas: date_fully_paid − due_date then .shift() per student group.

---

## Step 3.0: Student Information

**Source**: `src/modules/feature_engineering/credit_sales_machine_learning.py → FeatureEngineer`  
**Function**: `DemographicFeatureEngineer (internal)`

**Inputs**:
- Enrollee file (school_year, installment plan, student ID)

**Outputs**:
- plan_type (one-hot encoded)
- enrollment_streak
- plan_risk_score

**Notes**: Joins enrollee records to credit sales by pseudonymized student ID.

---

## Step 4.0: Payment History (Bracket Allocation)

**Source**: `src/modules/feature_engineering/credit_sales_machine_learning.py → InvoiceBuilder`  
**Function**: `_allocate_payment_brackets_sequential(df)`

**Inputs**:
- Payment amounts
- Due dates
- Payment dates

**Outputs**:
- paid_before_due_date
- paid_1_to_30_days
- paid_31_to_60_days
- paid_61_to_90_days
- paid_91_to_120_days
- paid_121_to_150_days
- paid_151_to_180_days
- paid_180_above_days
- remaining_accounts_receivables

**Notes**: Conditional bucketing: each payment → one bracket based on (payment_date − due_date) days elapsed.

---

## Step 5.0: Data Cleaning (CreditSalesProcessor)

**Source**: `src/modules/feature_engineering/credit_sales_machine_learning.py → CreditSalesProcessor`  
**Function**: `CreditSalesProcessor.show_data()`

**Inputs**:
- Raw revenue DataFrame
- Raw enrollee DataFrame

**Outputs**:
- df_data: ML training stream (censor==1, survival columns dropped)
- df_data_surv: Survival analysis stream (full, survival columns intact)
- plan_risk_map_cache.pkl: Fitted categorical risk mappings

**Notes**: Orchestrates three sub-classes: InvoiceBuilder (discount/adjustment/payment allocation, ThreadPool), FeatureEngineer (DTP lags, cumulative balances, plan encoding), InvoicePostProcessor (year filter, winsorization, column dropping). Splits output into two distinct data streams after filtering.

---

## Step 6.0: Data Preparation

**Source**: `src/modules/machine_learning/utils/data/data_preparation.py → DataPreparer`  
**Function**: `DataPreparer.prep_data(strategy, threshold)`

**Inputs**:
- df_data (ML stream)
- balance_strategy
- threshold (for HybridBalance)

**Outputs**:
- X_train, X_test (normalized)
- y_train, y_test (encoded)
- Resampled training set

**Notes**: Pipeline: encode_labels() → train_test_split() → resample() → normalize(). Balancing strategies: SMOTE, BorderlineSMOTE, SMOTEENN, SMOTETomek, HybridBalance. Optional LDA (apply_lda()) for dimensionality reduction to 4 components.

---

## Step 7.0: Data Partitioning

**Source**: `src/modules/machine_learning/utils/data/data_partitioning.py`  
**Function**: `data_partitioning_by_due_date(df, test_size)`

**Inputs**:
- df_data
- test_size (default 0.2)

**Outputs**:
- X_train, X_test (temporal split)
- y_train, y_test

**Notes**: Temporal split: invoices sorted by due_date; 80% earliest→train, 20% latest→test. No leakage.

---

## Step 7.5: Cox Survival Analysis Tuning [NEW — missing from DFD] ⚠️ **MISSING FROM DFD**

**Source**: `src/modules/machine_learning/utils/survival/cox_hyperparameter_tuner.py → CoxHyperparameterTuner`  
**Function**: `tune_cox_model(df_data_surv, results_root)`

**Inputs**:
- df_data_surv (survival stream: event indicator E, event time T)
- alpha_grid = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
- l1_ratios = [0.5, 0.75, 1.0]

**Outputs**:
- best_c_index: float (Harrell's concordance)
- best_surv_parameters: {alpha, l1_ratio}
- best_time_points: list of 9 optimal evaluation time points
- cox_tuning_report.xlsx: full grid search results

**Notes**: Grid search over 18 combos (6 alpha × 3 l1_ratio). K-Fold CV; scored by C-index per fold. Time points derived via get_slope_timepoints(T, E, n_points=9). This is a standalone phase between data partitioning and model building.

---

## Step 8.0: Model Building

**Source**: `src/modules/machine_learning/utils/training/run_models_parallel.py → SurvivalExperimentRunner`  
**Function**: `SurvivalExperimentRunner.run()`

**Inputs**:
- X_train/X_test (baseline)
- X_surv_train/X_surv_test (enhanced)
- Model configs
- balance strategies

**Outputs**:
- model_results_df: DataFrame with all metrics
- class_mappings_dict

**Notes**: Parallel execution (n_jobs=-1). Trains 15 model variants:
  8.1 Base classifiers (6): AdaBoost, RandomForest, XGBoost, DecisionTree, GaussianNB, KNN
  8.2 Ordinal classifiers (3): Ordinal-AdaBoost, Ordinal-RF, Ordinal-XGB
       Frank & Hall (2001): K−1=3 binary classifiers: P(y>0), P(y>1), P(y>2)
  8.3 Two-stage ensembles (6): XGB→XGB, XGB→RF, RF→RF, XGB→Ada, RF→Ada, Ada→XGB
       Stage 1: binary (on-time vs late), Stage 2: multiclass (30/60/90-day) on late-only subset
       Combined via chain rule: P(k|k>0) = P(late) × P(class=k|late)
  Tree-based restriction for 8.2/8.3: SelectFromModel requires feature_importances_,
  which only tree-based models expose (KNN/GNB excluded).

---

## Step 8.5: Survival Feature Generation [NEW — missing from DFD] ⚠️ **MISSING FROM DFD**

**Source**: `src/modules/machine_learning/utils/features/generate_survival_features.py`  
**Function**: `generate_survival_features(X_surv, T, E, X_train_raw, X_test_raw, best_params, time_points)`

**Inputs**:
- df_data_surv training/test splits
- best_surv_parameters from Cox tuning
- best_time_points (9 values)

**Outputs**:
- X_surv_train: original features + 9-point S(t) + 9-point H(t) + risk_score + E[T]
- X_surv_test: same enhanced structure for test set

**Notes**: Fits CoxnetSurvivalAnalysis on training survival data only (no leakage). Per sample at each of 9 time points: survival probability S(t), cumulative hazard H(t). Per sample: risk_score = exp(log_hazard_ratio), expected survival time E[T]. _safe_scale() clips to [-10,+10] before exp to prevent overflow. Creates a separate 'enhanced' dataset alongside the original 'baseline' dataset.

---

## Step 9.0: Feature Selection

**Source**: `src/modules/machine_learning/models/base_pipeline.py → BasePipeline.fit()`  
**Function**: `BasePipeline.fit(use_feature_selection=True)`

**Inputs**:
- X_train (baseline or enhanced)
- y_train
- fitted estimator

**Outputs**:
- Selected feature mask
- Reduced X_train, X_test

**Notes**: Tree models: SelectFromModel with MDI importance threshold. Stacked ensemble: permutation importance (no built-in feature_importances_). KNN/GNB: custom influence score or permutation importance.

---

## Step 10.0: Model Execution

**Source**: `src/modules/machine_learning/models/base_pipeline.py → BasePipeline.evaluate()`  
**Function**: `BasePipeline.evaluate(X_test, y_test)`

**Inputs**:
- X_test (selected features)
- y_test

**Outputs**:
- y_pred
- y_proba (class probabilities)

**Notes**: Generates predictions on test set using fitted model (baseline and enhanced separately).

---

## Step 11.0: Model Evaluation

**Source**: `src/modules/machine_learning/utils/training/data_evaluation.py`  
**Function**: `data_evaluation(y_true, y_pred, y_proba)`

**Inputs**:
- y_true
- y_pred
- y_proba

**Outputs**:
- accuracy, precision, recall, F1 (macro)
- ROC-AUC (macro OvR)
- confusion matrix
- ROC/PR curves

**Notes**: Multiclass-aware: computes per-class ROC and PR curves for 4-class ordinal problem.

---

## Step 12.0: Model Combination (Stacking + Comparison)

**Source**: `src/modules/machine_learning/models/stacked_ensemble.py → StackedEnsemblePipeline`  
**Function**: `StackedEnsemblePipeline.fit()`

**Inputs**:
- X_train
- y_train
- Base estimators (AdaBoost, RF, GradientBoosting, Bagging)

**Outputs**:
- StackingClassifier fitted
- Meta-learner (LogisticRegression) fitted

**Notes**: Sklearn StackingClassifier with 4 base learners + LogisticRegression meta-learner. Feature selection via permutation importance on full ensemble. Results stored in SQLite (results.db) for Step 4 comparison dashboard.

---

## Step 13.0: Classify Transactions (Inference)

**Source**: `src/app/screens/initial_setup/callbacks/step_5.py`  
**Function**: `run_finalization() → InferencePipeline`

**Inputs**:
- New credit sales data
- Selected model config
- plan_risk_map_cache.pkl

**Outputs**:
- Credit risk label (0/1/2/3)
- Class probabilities
- finalized_{model_key}.pkl

**Notes**: Step 5: Re-trains selected model on full dataset (no test split). Re-fits Cox survival model on full data. Bundles InferencePipeline with all preprocessing for deployment.

---

## Step 14.0: Generate Summaries

**Source**: `src/app/screens/ (Step 4 comparison dashboard)`  
**Function**: `comparative_model_dashboard callbacks`

**Inputs**:
- results.db (SQLite)
- model_results_df

**Outputs**:
- Model leaderboard table
- Collection forecasts
- Summary statistics

**Notes**: Step 4 dashboard aggregates all model results from SQLite for comparison.

---

## Step 15.0: Payment Analysis

**Source**: `src/app/screens/ (Step 4 dashboard)`  
**Function**: `Payment analysis callbacks`

**Inputs**:
- Credit risk predictions
- Invoice due dates
- Payment history

**Outputs**:
- Payment timing analysis
- Aging analysis
- Collection forecast

**Notes**: Analysis of predicted payment behavior: distribution across 30/60/90-day buckets.

---

## Step 16.0: Visualization

**Source**: `src/app/screens/ (Step 4 dashboard)`  
**Function**: `Visualization callbacks`

**Inputs**:
- Model results
- Feature importances
- Confusion matrices

**Outputs**:
- Model comparison charts
- Feature importance plots
- ROC/PR curve plots

**Notes**: Interactive Dash visualizations for model selection and feature analysis.

---
