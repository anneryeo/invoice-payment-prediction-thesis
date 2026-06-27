"""
Phase 3.1: Compare the diagram pipeline map (Phase 2.1) against the
code pipeline trace (Phase 2.2) and produce discrepancy_analysis.md.

Discrepancy categories:
  - IN_SYNC: DFD process matches code step
  - MISSING_PROCESS: Code step has no DFD process
  - MISSING_FLOW: Code data flow not shown in DFD
  - MISLABELED: DFD label doesn't fully reflect code reality
  - NOTE_ONLY: Discrepancy noted but out of scope for this revision
"""

from pathlib import Path
from dataclasses import dataclass


OUTPUT = Path(__file__).parents[1] / "outputs" / "discrepancy_analysis.md"


@dataclass
class Discrepancy:
    id: str
    category: str    # IN_SYNC | MISSING_PROCESS | MISSING_FLOW | MISLABELED | NOTE_ONLY
    dfd: str         # Which DFD (Processing / Modelling / Analysis / All)
    process: str     # Process number(s)
    description: str
    code_evidence: str
    resolution: str  # IMPLEMENT | NOTE_ONLY | DEFERRED
    priority: str    # HIGH | MEDIUM | LOW


DISCREPANCIES: list[Discrepancy] = [
    # ─── IN SYNC ───────────────────────────────────────────────────────────────
    Discrepancy("S1", "IN_SYNC", "Processing", "1.0",
        "Data importation maps to base64 Excel decode + Revenues async loader",
        "step_3.py:clean_datasets() → calamine engine; Revenues class async parallel load",
        "NOTE_ONLY", "LOW"),
    Discrepancy("S2", "IN_SYNC", "Processing", "2.0",
        "Time to payment maps to DTP lag feature computation",
        "FeatureEngineer._merge_dtp(): date_fully_paid − due_date + .shift() per student group",
        "NOTE_ONLY", "LOW"),
    Discrepancy("S3", "IN_SYNC", "Processing", "3.0",
        "Student information maps to demographic feature engineering",
        "DemographicFeatureEngineer: plan_type, enrollment_streak, plan_risk_score",
        "NOTE_ONLY", "LOW"),
    Discrepancy("S4", "IN_SYNC", "Processing", "4.0",
        "Payment history maps to 30/60/90 day bracket allocation",
        "_allocate_payment_brackets_sequential(): 8 time-bucketed amount columns",
        "NOTE_ONLY", "LOW"),
    Discrepancy("S5", "IN_SYNC", "Processing", "5.0",
        "Data cleaning maps to CreditSalesProcessor orchestration",
        "CreditSalesProcessor: InvoiceBuilder + FeatureEngineer + InvoicePostProcessor",
        "NOTE_ONLY", "LOW"),
    Discrepancy("S6", "IN_SYNC", "Modelling", "6.0",
        "Data preparation maps to DataPreparer (encode → split → resample → normalize)",
        "DataPreparer.prep_data(): SMOTE variants, StandardScaler, optional LDA",
        "NOTE_ONLY", "LOW"),
    Discrepancy("S7", "IN_SYNC", "Modelling", "7.0",
        "Data partitioning maps to temporal train/test split",
        "data_partitioning_by_due_date(): sort by due_date, 80/20 temporal cutoff",
        "NOTE_ONLY", "LOW"),
    Discrepancy("S8", "IN_SYNC", "Modelling", "8.0",
        "Model building maps to SurvivalExperimentRunner parallel training",
        "run_models_parallel.py: 15 model variants across 3 categories (base, ordinal, two-stage)",
        "NOTE_ONLY", "LOW"),
    Discrepancy("S9", "IN_SYNC", "Modelling", "9.0–12.0",
        "Feature selection, model execution, evaluation, combination all present",
        "base_pipeline.py, data_evaluation.py, stacked_ensemble.py",
        "NOTE_ONLY", "LOW"),
    Discrepancy("S10", "IN_SYNC", "Analysis", "13.0–16.0",
        "Classify, summarize, payment analysis, visualization all present in Step 4/5",
        "step_5.py (finalization/inference), Step 4 dashboard callbacks",
        "NOTE_ONLY", "LOW"),

    # ─── MISSING PROCESSES ─────────────────────────────────────────────────────
    Discrepancy("D1", "MISSING_PROCESS", "Modelling", "7.5 (new)",
        "Cox Survival Analysis Tuning is a major standalone phase with no DFD process",
        (
            "tune_cox_model(): CoxHyperparameterTuner grid search 18 combos (6 alpha × 3 l1_ratio). "
            "K-Fold CV; scored by Harrell C-index. Outputs best_surv_parameters, best_time_points (9). "
            "This runs AFTER data partitioning and BEFORE model building — a dedicated phase."
        ),
        "IMPLEMENT", "HIGH"),
    Discrepancy("D2", "MISSING_PROCESS", "Modelling", "8.5 (new)",
        "Survival Feature Generation creates a separate 'enhanced' dataset not shown in DFD",
        (
            "generate_survival_features(): fits CoxnetSurvivalAnalysis on training data. "
            "Computes S(t), H(t) at 9 time points, risk_score, E[T] per sample. "
            "Concatenates with original features → enhanced dataset distinct from baseline."
        ),
        "IMPLEMENT", "HIGH"),

    # ─── MISSING DATA FLOWS ────────────────────────────────────────────────────
    Discrepancy("D3", "MISSING_FLOW", "Processing", "5.0",
        "After Data cleaning (5.0), code produces two distinct output streams; DFD shows only one",
        (
            "clean_datasets() returns df_data (censor==1, survival cols dropped) AND "
            "df_data_surv (full, survival cols intact). The DFD shows a single 'Cleaned data' output. "
            "Both streams have distinct downstream consumers: baseline ML vs Cox tuning."
        ),
        "IMPLEMENT", "HIGH"),
    Discrepancy("D4", "MISSING_FLOW", "Modelling", "6.0",
        "Data preparation handles SMOTE balancing but DFD does not label it as such",
        (
            "DataPreparer.resample() runs SMOTE, BorderlineSMOTE, SMOTEENN, SMOTETomek, or HybridBalance "
            "as a required pre-training step. The 'Model hyperparameters' edge into 6.0 should "
            "include 'balancing strategy' as an explicit input label."
        ),
        "IMPLEMENT", "MEDIUM"),
    Discrepancy("D5", "MISSING_FLOW", "Modelling", "8.0",
        "Model building receives both baseline and enhanced feature sets; DFD shows only one data flow in",
        (
            "SurvivalExperimentRunner trains each model TWICE: once with baseline features (X_train) "
            "and once with survival-augmented features (X_surv_train). The DFD only shows a single "
            "'Training data' input to Model building."
        ),
        "IMPLEMENT", "HIGH"),

    # ─── MISLABELED ────────────────────────────────────────────────────────────
    Discrepancy("D0", "MISLABELED", "Modelling", "8.0",
        "Process 8.0 'Model building' label understates the 15-variant architecture inside it",
        (
            "8.0 covers: 6 base classifiers (Ada, RF, XGB, DT, GNB, KNN), "
            "3 ordinal classifiers (Ordinal-Ada/RF/XGB via Frank & Hall K-1 decomposition), "
            "6 two-stage ensembles (XGB/RF/Ada × XGB/RF/Ada combinations). "
            "Sub-DFD 8.0 needed to show the three sub-categories."
        ),
        "IMPLEMENT", "HIGH"),

    # ─── NOTE ONLY ─────────────────────────────────────────────────────────────
    Discrepancy("N0", "NOTE_ONLY", "Modelling", "8.0",
        "Chapter 3 Section 3.5 narrative on 8.0 needs to include the 15-variant model table",
        (
            "Section 3.5 narrates individual models but may not clearly state the full 15-variant grid "
            "(6 base + 3 ordinal + 6 two-stage), the feature_importances_ restriction for ordinal/two-stage, "
            "and the Frank & Hall decomposition details."
        ),
        "NOTE_ONLY", "HIGH"),
    Discrepancy("N1", "NOTE_ONLY", "Modelling", "6.0–12.0",
        "Modelling DFD narrative does not mention Cox tuning as a dedicated phase",
        "Sections 3.4 and 3.6 discuss survival analysis but the DFD narrative may not name process 7.5.",
        "NOTE_ONLY", "MEDIUM"),
    Discrepancy("N2", "NOTE_ONLY", "Processing", "5.0",
        "Section 3.2 narrative for 5.0 should describe the data stream split",
        "CreditSalesProcessor output split into df_data (ML) and df_data_surv (survival) not described.",
        "NOTE_ONLY", "MEDIUM"),
    Discrepancy("N3", "NOTE_ONLY", "Analysis", "—",
        "Step 5 model finalization (re-train on full dataset) not shown in Analysis DFD",
        "step_5.py: re-trains selected model on full data, saves finalized_*.pkl InferencePipeline.",
        "NOTE_ONLY", "LOW"),
    Discrepancy("N4", "NOTE_ONLY", "All", "—",
        "Conceptual Framework diagram's Modelling box doesn't reflect Cox as a sub-component",
        "High-level diagram shows Pre-Processing / Modelling / Analysis but Cox tuning is absent.",
        "NOTE_ONLY", "LOW"),
    Discrepancy("N5", "NOTE_ONLY", "All", "—",
        "Individual model hyperparameter configs in Chapter 3 may not match settings.json",
        "Model-specific hyperparameters (e.g., n_estimators, max_depth) described in 3.5 vs settings.",
        "NOTE_ONLY", "LOW"),
]


class DiscrepancyAnalyzer:
    def analyze(self) -> list[Discrepancy]:
        return DISCREPANCIES

    def to_markdown(self, items: list[Discrepancy]) -> str:
        lines = ["# Discrepancy Analysis: DFD vs Code Pipeline\n"]
        lines.append("_Compares Level-1 DFD elements (Phase 2.1) against the actual code pipeline (Phase 2.2)._\n")

        by_cat = {}
        for d in items:
            by_cat.setdefault(d.category, []).append(d)

        cat_order = ["IN_SYNC", "MISSING_PROCESS", "MISSING_FLOW", "MISLABELED", "NOTE_ONLY"]
        cat_labels = {
            "IN_SYNC": "✅ In Sync",
            "MISSING_PROCESS": "🔴 Missing DFD Processes (IMPLEMENT)",
            "MISSING_FLOW": "🟠 Missing Data Flows (IMPLEMENT)",
            "MISLABELED": "🟡 Mislabeled / Underspecified (IMPLEMENT)",
            "NOTE_ONLY": "📝 Noted Only — Out of Scope for This Revision",
        }

        for cat in cat_order:
            if cat not in by_cat:
                continue
            lines.append(f"## {cat_labels[cat]}\n")
            lines.append("| ID | DFD | Process | Priority | Description |")
            lines.append("|----|-----|---------|----------|-------------|")
            for d in by_cat[cat]:
                lines.append(f"| {d.id} | {d.dfd} | {d.process} | {d.priority} | {d.description} |")
            lines.append("")

        lines.append("## Detailed Findings\n")
        implement = [d for d in items if d.resolution == "IMPLEMENT"]
        for d in implement:
            lines.append(f"### {d.id} — {d.description}\n")
            lines.append(f"**DFD**: {d.dfd}  |  **Process**: {d.process}  |  **Priority**: {d.priority}\n")
            lines.append(f"**Code evidence**: {d.code_evidence}\n")
            lines.append("")

        return "\n".join(lines)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    analyzer = DiscrepancyAnalyzer()
    items = analyzer.analyze()
    content = analyzer.to_markdown(items)
    OUTPUT.write_text(content, encoding="utf-8")
    implement = [d for d in items if d.resolution == "IMPLEMENT"]
    note_only = [d for d in items if d.resolution == "NOTE_ONLY"]
    print(f"Written to {OUTPUT}")
    print(f"  IMPLEMENT: {len(implement)} items  |  NOTE_ONLY: {len(note_only)} items")


if __name__ == "__main__":
    main()
