"""
Phase 2.2: Document the actual code pipeline starting from step_3.py
and produce code_pipeline_trace.md — a structured reference of each step,
its source file, key function, inputs, and outputs.
"""

from pathlib import Path
from dataclasses import dataclass, field


OUTPUT = Path(__file__).parents[1] / "outputs" / "code_pipeline_trace.md"
SRC_ROOT = Path(__file__).parents[4] / "src"


@dataclass
class PipelineStep:
    number: str
    name: str
    source_file: str
    function: str
    inputs: list[str]
    outputs: list[str]
    notes: str = ""


PIPELINE_STEPS: list[PipelineStep] = [
    PipelineStep(
        number="1.0",
        name="Data Importation",
        source_file="src/app/screens/initial_setup/callbacks/step_3.py → clean_datasets()",
        function="clean_datasets(revenues_content, enrollees_content)",
        inputs=["Revenue Ledger (.xlsx) — base64-encoded", "Enrollee Information (.xlsx) — base64-encoded"],
        outputs=["Raw revenue DataFrame", "Raw enrollee DataFrame"],
        notes="Base64 decoding + Excel read via calamine engine. Async parallel file loads in Revenues loader.",
    ),
    PipelineStep(
        number="2.0",
        name="Time to Payment (DTP)",
        source_file="src/modules/feature_engineering/credit_sales_machine_learning.py → FeatureEngineer._merge_dtp()",
        function="_merge_dtp(df)",
        inputs=["Matched payment records", "Due dates per receivable"],
        outputs=["DTP lag features: dtp_1, dtp_2, dtp_3, dtp_4 (days from due to fully paid, per prior invoice)"],
        notes="Vectorized pandas: date_fully_paid − due_date then .shift() per student group.",
    ),
    PipelineStep(
        number="3.0",
        name="Student Information",
        source_file="src/modules/feature_engineering/credit_sales_machine_learning.py → FeatureEngineer",
        function="DemographicFeatureEngineer (internal)",
        inputs=["Enrollee file (school_year, installment plan, student ID)"],
        outputs=["plan_type (one-hot encoded)", "enrollment_streak", "plan_risk_score"],
        notes="Joins enrollee records to credit sales by pseudonymized student ID.",
    ),
    PipelineStep(
        number="4.0",
        name="Payment History (Bracket Allocation)",
        source_file="src/modules/feature_engineering/credit_sales_machine_learning.py → InvoiceBuilder",
        function="_allocate_payment_brackets_sequential(df)",
        inputs=["Payment amounts", "Due dates", "Payment dates"],
        outputs=[
            "paid_before_due_date", "paid_1_to_30_days", "paid_31_to_60_days",
            "paid_61_to_90_days", "paid_91_to_120_days", "paid_121_to_150_days",
            "paid_151_to_180_days", "paid_180_above_days", "remaining_accounts_receivables",
        ],
        notes="Conditional bucketing: each payment → one bracket based on (payment_date − due_date) days elapsed.",
    ),
    PipelineStep(
        number="5.0",
        name="Data Cleaning (CreditSalesProcessor)",
        source_file="src/modules/feature_engineering/credit_sales_machine_learning.py → CreditSalesProcessor",
        function="CreditSalesProcessor.show_data()",
        inputs=["Raw revenue DataFrame", "Raw enrollee DataFrame"],
        outputs=[
            "df_data: ML training stream (censor==1, survival columns dropped)",
            "df_data_surv: Survival analysis stream (full, survival columns intact)",
            "plan_risk_map_cache.pkl: Fitted categorical risk mappings",
        ],
        notes=(
            "Orchestrates three sub-classes: InvoiceBuilder (discount/adjustment/payment allocation, ThreadPool), "
            "FeatureEngineer (DTP lags, cumulative balances, plan encoding), "
            "InvoicePostProcessor (year filter, winsorization, column dropping). "
            "Splits output into two distinct data streams after filtering."
        ),
    ),
    PipelineStep(
        number="6.0",
        name="Data Preparation",
        source_file="src/modules/machine_learning/utils/data/data_preparation.py → DataPreparer",
        function="DataPreparer.prep_data(strategy, threshold)",
        inputs=["df_data (ML stream)", "balance_strategy", "threshold (for HybridBalance)"],
        outputs=["X_train, X_test (normalized)", "y_train, y_test (encoded)", "Resampled training set"],
        notes=(
            "Pipeline: encode_labels() → train_test_split() → resample() → normalize(). "
            "Balancing strategies: SMOTE, BorderlineSMOTE, SMOTEENN, SMOTETomek, HybridBalance. "
            "Optional LDA (apply_lda()) for dimensionality reduction to 4 components."
        ),
    ),
    PipelineStep(
        number="7.0",
        name="Data Partitioning",
        source_file="src/modules/machine_learning/utils/data/data_partitioning.py",
        function="data_partitioning_by_due_date(df, test_size)",
        inputs=["df_data", "test_size (default 0.2)"],
        outputs=["X_train, X_test (temporal split)", "y_train, y_test"],
        notes="Temporal split: invoices sorted by due_date; 80% earliest→train, 20% latest→test. No leakage.",
    ),
    PipelineStep(
        number="7.5",
        name="Cox Survival Analysis Tuning [NEW — missing from DFD]",
        source_file="src/modules/machine_learning/utils/survival/cox_hyperparameter_tuner.py → CoxHyperparameterTuner",
        function="tune_cox_model(df_data_surv, results_root)",
        inputs=[
            "df_data_surv (survival stream: event indicator E, event time T)",
            "alpha_grid = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]",
            "l1_ratios = [0.5, 0.75, 1.0]",
        ],
        outputs=[
            "best_c_index: float (Harrell's concordance)",
            "best_surv_parameters: {alpha, l1_ratio}",
            "best_time_points: list of 9 optimal evaluation time points",
            "cox_tuning_report.xlsx: full grid search results",
        ],
        notes=(
            "Grid search over 18 combos (6 alpha × 3 l1_ratio). "
            "K-Fold CV; scored by C-index per fold. "
            "Time points derived via get_slope_timepoints(T, E, n_points=9). "
            "This is a standalone phase between data partitioning and model building."
        ),
    ),
    PipelineStep(
        number="8.0",
        name="Model Building",
        source_file="src/modules/machine_learning/utils/training/run_models_parallel.py → SurvivalExperimentRunner",
        function="SurvivalExperimentRunner.run()",
        inputs=["X_train/X_test (baseline)", "X_surv_train/X_surv_test (enhanced)", "Model configs", "balance strategies"],
        outputs=["model_results_df: DataFrame with all metrics", "class_mappings_dict"],
        notes=(
            "Parallel execution (n_jobs=-1). Trains 15 model variants:\n"
            "  8.1 Base classifiers (6): AdaBoost, RandomForest, XGBoost, DecisionTree, GaussianNB, KNN\n"
            "  8.2 Ordinal classifiers (3): Ordinal-AdaBoost, Ordinal-RF, Ordinal-XGB\n"
            "       Frank & Hall (2001): K−1=3 binary classifiers: P(y>0), P(y>1), P(y>2)\n"
            "  8.3 Two-stage ensembles (6): XGB→XGB, XGB→RF, RF→RF, XGB→Ada, RF→Ada, Ada→XGB\n"
            "       Stage 1: binary (on-time vs late), Stage 2: multiclass (30/60/90-day) on late-only subset\n"
            "       Combined via chain rule: P(k|k>0) = P(late) × P(class=k|late)\n"
            "  Tree-based restriction for 8.2/8.3: SelectFromModel requires feature_importances_,\n"
            "  which only tree-based models expose (KNN/GNB excluded)."
        ),
    ),
    PipelineStep(
        number="8.5",
        name="Survival Feature Generation [NEW — missing from DFD]",
        source_file="src/modules/machine_learning/utils/features/generate_survival_features.py",
        function="generate_survival_features(X_surv, T, E, X_train_raw, X_test_raw, best_params, time_points)",
        inputs=[
            "df_data_surv training/test splits",
            "best_surv_parameters from Cox tuning",
            "best_time_points (9 values)",
        ],
        outputs=[
            "X_surv_train: original features + 9-point S(t) + 9-point H(t) + risk_score + E[T]",
            "X_surv_test: same enhanced structure for test set",
        ],
        notes=(
            "Fits CoxnetSurvivalAnalysis on training survival data only (no leakage). "
            "Per sample at each of 9 time points: survival probability S(t), cumulative hazard H(t). "
            "Per sample: risk_score = exp(log_hazard_ratio), expected survival time E[T]. "
            "_safe_scale() clips to [-10,+10] before exp to prevent overflow. "
            "Creates a separate 'enhanced' dataset alongside the original 'baseline' dataset."
        ),
    ),
    PipelineStep(
        number="9.0",
        name="Feature Selection",
        source_file="src/modules/machine_learning/models/base_pipeline.py → BasePipeline.fit()",
        function="BasePipeline.fit(use_feature_selection=True)",
        inputs=["X_train (baseline or enhanced)", "y_train", "fitted estimator"],
        outputs=["Selected feature mask", "Reduced X_train, X_test"],
        notes=(
            "Tree models: SelectFromModel with MDI importance threshold. "
            "Stacked ensemble: permutation importance (no built-in feature_importances_). "
            "KNN/GNB: custom influence score or permutation importance."
        ),
    ),
    PipelineStep(
        number="10.0",
        name="Model Execution",
        source_file="src/modules/machine_learning/models/base_pipeline.py → BasePipeline.evaluate()",
        function="BasePipeline.evaluate(X_test, y_test)",
        inputs=["X_test (selected features)", "y_test"],
        outputs=["y_pred", "y_proba (class probabilities)"],
        notes="Generates predictions on test set using fitted model (baseline and enhanced separately).",
    ),
    PipelineStep(
        number="11.0",
        name="Model Evaluation",
        source_file="src/modules/machine_learning/utils/training/data_evaluation.py",
        function="data_evaluation(y_true, y_pred, y_proba)",
        inputs=["y_true", "y_pred", "y_proba"],
        outputs=["accuracy, precision, recall, F1 (macro)", "ROC-AUC (macro OvR)", "confusion matrix", "ROC/PR curves"],
        notes="Multiclass-aware: computes per-class ROC and PR curves for 4-class ordinal problem.",
    ),
    PipelineStep(
        number="12.0",
        name="Model Combination (Stacking + Comparison)",
        source_file="src/modules/machine_learning/models/stacked_ensemble.py → StackedEnsemblePipeline",
        function="StackedEnsemblePipeline.fit()",
        inputs=["X_train", "y_train", "Base estimators (AdaBoost, RF, GradientBoosting, Bagging)"],
        outputs=["StackingClassifier fitted", "Meta-learner (LogisticRegression) fitted"],
        notes=(
            "Sklearn StackingClassifier with 4 base learners + LogisticRegression meta-learner. "
            "Feature selection via permutation importance on full ensemble. "
            "Results stored in SQLite (results.db) for Step 4 comparison dashboard."
        ),
    ),
    PipelineStep(
        number="13.0",
        name="Classify Transactions (Inference)",
        source_file="src/app/screens/initial_setup/callbacks/step_5.py",
        function="run_finalization() → InferencePipeline",
        inputs=["New credit sales data", "Selected model config", "plan_risk_map_cache.pkl"],
        outputs=["Credit risk label (0/1/2/3)", "Class probabilities", "finalized_{model_key}.pkl"],
        notes=(
            "Step 5: Re-trains selected model on full dataset (no test split). "
            "Re-fits Cox survival model on full data. "
            "Bundles InferencePipeline with all preprocessing for deployment."
        ),
    ),
    PipelineStep(
        number="14.0",
        name="Generate Summaries",
        source_file="src/app/screens/ (Step 4 comparison dashboard)",
        function="comparative_model_dashboard callbacks",
        inputs=["results.db (SQLite)", "model_results_df"],
        outputs=["Model leaderboard table", "Collection forecasts", "Summary statistics"],
        notes="Step 4 dashboard aggregates all model results from SQLite for comparison.",
    ),
    PipelineStep(
        number="15.0",
        name="Payment Analysis",
        source_file="src/app/screens/ (Step 4 dashboard)",
        function="Payment analysis callbacks",
        inputs=["Credit risk predictions", "Invoice due dates", "Payment history"],
        outputs=["Payment timing analysis", "Aging analysis", "Collection forecast"],
        notes="Analysis of predicted payment behavior: distribution across 30/60/90-day buckets.",
    ),
    PipelineStep(
        number="16.0",
        name="Visualization",
        source_file="src/app/screens/ (Step 4 dashboard)",
        function="Visualization callbacks",
        inputs=["Model results", "Feature importances", "Confusion matrices"],
        outputs=["Model comparison charts", "Feature importance plots", "ROC/PR curve plots"],
        notes="Interactive Dash visualizations for model selection and feature analysis.",
    ),
]


class PipelineTracer:
    def trace(self) -> list[PipelineStep]:
        return PIPELINE_STEPS

    def to_markdown(self, steps: list[PipelineStep]) -> str:
        lines = ["# Code Pipeline Trace\n"]
        lines.append("_Source of truth: step_3.py call graph and module exploration._\n")
        lines.append("Steps marked **[NEW]** are in the code but absent from the current Level-1 DFDs.\n")

        for step in steps:
            is_new = "[NEW" in step.name
            marker = " ⚠️ **MISSING FROM DFD**" if is_new else ""
            lines.append(f"## Step {step.number}: {step.name}{marker}\n")
            lines.append(f"**Source**: `{step.source_file}`  ")
            lines.append(f"**Function**: `{step.function}`\n")

            lines.append("**Inputs**:")
            for inp in step.inputs:
                lines.append(f"- {inp}")
            lines.append("")

            lines.append("**Outputs**:")
            for out in step.outputs:
                lines.append(f"- {out}")
            lines.append("")

            if step.notes:
                lines.append(f"**Notes**: {step.notes}\n")

            lines.append("---\n")

        return "\n".join(lines)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tracer = PipelineTracer()
    steps = tracer.trace()
    content = tracer.to_markdown(steps)
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"Written {len(steps)} pipeline steps to {OUTPUT}")
    new_steps = [s for s in steps if "[NEW" in s.name]
    print(f"\nSteps missing from DFDs ({len(new_steps)}):")
    for s in new_steps:
        print(f"  {s.number}: {s.name}")


if __name__ == "__main__":
    main()
