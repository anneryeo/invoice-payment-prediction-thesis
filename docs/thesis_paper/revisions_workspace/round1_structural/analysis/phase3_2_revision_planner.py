"""Phase 3.2 — Revision Planner.

Reads the discrepancy analysis and produces a structured revision plan
covering:
  - Level-1 DFD changes already implemented (D1–D4)
  - Level-2 sub-DFD files generated
  - Narrative paragraph replacement text for each section
  - Noted-only items (N0–N5) with recommended future action
"""

from pathlib import Path
from datetime import date

SCRIPT_DIR   = Path(__file__).parent
OUTPUTS_DIR  = SCRIPT_DIR.parent / "outputs"
DISC_PATH    = OUTPUTS_DIR / "discrepancy_analysis.md"
OUTPUT_PATH  = OUTPUTS_DIR / "revision_plan.md"

# ─────────────────────────────────────────────────────────────────────────────
# Narrative replacement blocks
# ─────────────────────────────────────────────────────────────────────────────

NARRATIVES = {

    # ── D3 ── Processing DFD §3.3.2 (Data cleaning outputs)
    "D3_processing_5_0": {
        "section": "3.3.2 — Data Cleaning (Processing Component DFD)",
        "heading":  "Process 5.0 Output Data Flows",
        "original": (
            'The cleaned data output from Process 5.0 is passed to the modelling '
            'component for further preparation.'
        ),
        "replacement": (
            'After cleaning, the pipeline produces two distinct data streams. '
            'The first stream — referred to as ML training data — contains only '
            'records where the censor indicator equals one (i.e., the event was '
            'observed). Survival-specific columns are dropped from this stream '
            'before it is forwarded to the modelling component for balancing, '
            'partitioning, and classifier training. '
            'The second stream — referred to as survival analysis data — retains '
            'all records and all survival columns (time-to-event T and event '
            'indicator E). This stream is forwarded exclusively to the Cox '
            'Proportional Hazards tuning sub-process (7.5) and subsequently to '
            'the survival feature generation sub-process (8.5).'
        ),
    },

    # ── D1 ── Modelling DFD §3.4.x (Cox tuning — new process 7.5)
    "D1_modelling_7_5": {
        "section": "3.4.x — Cox Survival Analysis Tuning (Modelling Component DFD)",
        "heading":  "Process 7.5 — Cox Survival Analysis Tuning [NEW]",
        "original": (
            "(No corresponding narrative existed; this process was absent from "
            "the Chapter 3 DFD and manuscript.)"
        ),
        "replacement": (
            'Process 7.5 performs hyperparameter tuning for the Cox Proportional '
            'Hazards model (CoxnetSurvivalAnalysis, scikit-survival) using the '
            'survival analysis data stream produced by Process 5.0. '
            'A grid of 18 hyperparameter combinations is evaluated: six values '
            'of the regularisation strength alpha and three values of the '
            'elastic-net mixing parameter l1_ratio. '
            'For each combination, k-fold cross-validation is applied, and each '
            'fold is scored using Harrell\'s concordance index (C-index). The '
            'combination that maximises the mean C-index across folds is selected '
            'as the best set of hyperparameters. '
            'Following hyperparameter selection, nine optimal time points are '
            'derived from the observed event distribution using a slope-change '
            'algorithm applied to the Kaplan-Meier survival curve. These nine '
            'time points are stored alongside the best hyperparameters and are '
            'used by Process 8.5 to generate per-sample survival features.'
        ),
    },

    # ── D2 ── Modelling DFD §3.4.x (Survival feature generation — new process 8.5)
    "D2_modelling_8_5": {
        "section": "3.4.x — Survival Feature Generation (Modelling Component DFD)",
        "heading":  "Process 8.5 — Survival Feature Generation [NEW]",
        "original": (
            "(No corresponding narrative existed; this process was absent from "
            "the Chapter 3 DFD and manuscript.)"
        ),
        "replacement": (
            'Process 8.5 generates survival-derived features that augment the '
            'original feature set before classifier training. '
            'Using the best hyperparameters obtained from Process 7.5, a '
            'CoxnetSurvivalAnalysis model is fitted on the training split of the '
            'survival analysis data. The fitted model is then applied to both '
            'training and test splits to compute, for each sample, the following '
            'features at each of the nine optimal time points: the survival '
            'probability S(t) and the cumulative hazard H(t). Additionally, a '
            'scalar risk score and the expected survival time E[T] are computed '
            'for each sample. '
            'These survival-derived features are concatenated with the original '
            'feature matrix to produce an enhanced dataset. Both the original '
            'and enhanced datasets are forwarded to Process 8.0 (Model Building), '
            'enabling a paired comparison between baseline and survival-augmented '
            'classification performance.'
        ),
    },

    # ── D4 ── Modelling DFD §3.4.1 (Data preparation — balancing annotation)
    "D4_modelling_6_0": {
        "section": "3.4.1 — Data Preparation (Modelling Component DFD)",
        "heading":  "Process 6.0 — Data Preparation (balancing annotation)",
        "original": (
            'The data preparation process encodes categorical labels and '
            'normalises numerical features before model training.'
        ),
        "replacement": (
            'Process 6.0 performs four sequential operations on the ML training '
            'data stream received from Process 5.0. '
            'First, ordinal labels are encoded: on-time, 30-day, 60-day, and '
            '90-day payment outcomes are mapped to integer classes 0–3. '
            'Second, a temporal train/test split is applied by sorting records '
            'on the instalment due date and assigning the earliest 80 percent '
            'of records to the training partition and the latest 20 percent to '
            'the test partition, preventing data leakage from future records. '
            'Third, a user-selected balancing strategy is applied to the training '
            'partition only. Five strategies are supported: SMOTE, Borderline '
            'SMOTE, SMOTEENN, SMOTETomek, and the custom HybridBalance strategy; '
            'each strategy is treated as a separate experimental condition. '
            'Fourth, min-max normalisation is applied to all numerical features. '
            'An optional linear discriminant analysis (LDA) projection may also '
            'be applied at this stage to reduce dimensionality before forwarding '
            'the prepared data to Process 7.0.'
        ),
    },

    # ── D0 / N0 ── Modelling DFD §3.4.4 (Model building — expanded)
    "D0_N0_modelling_8_0": {
        "section": "3.4.4 — Model Building (Modelling Component DFD)",
        "heading":  "Process 8.0 — Model Building [EXPANDED]",
        "original": (
            'The model building process trains the selected classifiers on the '
            'prepared training data.'
        ),
        "replacement": (
            'Process 8.0 trains fifteen classifiers organised into three '
            'sub-categories: six base classifiers, three ordinal classifiers, '
            'and six two-stage ensemble classifiers.'
            '\n\n'
            'BASE CLASSIFIERS (8.1). Six classifiers are trained independently '
            'on both the original and survival-enhanced feature sets: '
            'AdaBoost (AdaBoostClassifier), Random Forest (RandomForestClassifier), '
            'XGBoost (XGBClassifier), Decision Tree (DecisionTreeClassifier), '
            'Gaussian Naive Bayes (GaussianNB), and K-Nearest Neighbours '
            '(KNeighborsClassifier). '
            'Each classifier is trained under each of the five balancing '
            'strategies, yielding a full experimental matrix.'
            '\n\n'
            'ORDINAL CLASSIFIERS (8.2). Three ordinal classifiers are built '
            'using the Frank and Hall (2001) binary decomposition method: '
            'Ordinal AdaBoost, Ordinal Random Forest, and Ordinal XGBoost. '
            'For four payment outcome classes (0–3), the method trains '
            'K-1 = 3 binary classifiers: '
            'Classifier 0 learns P(y > 0) — on-time vs any late; '
            'Classifier 1 learns P(y > 1) — at most 30-day late vs 60-day or '
            'more late; '
            'Classifier 2 learns P(y > 2) — at most 60-day late vs 90-day late. '
            'Class probabilities are recovered as differences: '
            'P(class = k) = P(y > k-1) - P(y > k), with boundary values '
            'P(y > -1) = 1 and P(y > K-1) = 0. '
            'Monotonicity is enforced by clipping negative differences to zero '
            'and re-normalising the resulting distribution.'
            '\n\n'
            'TWO-STAGE ENSEMBLE CLASSIFIERS (8.3). Six two-stage ensemble '
            'classifiers are trained, each combining two tree-based estimators: '
            'XGBoost-XGBoost, XGBoost-Random Forest, Random Forest-Random Forest, '
            'XGBoost-AdaBoost, Random Forest-AdaBoost, and AdaBoost-XGBoost. '
            'Stage 1 is a binary classifier trained on the full dataset to '
            'predict P(late) — the probability that a payment is late (any class '
            'other than on-time). Stage 2 is a multiclass classifier trained '
            'only on the late subset, predicting P(class = k | late) for the '
            '30-day, 60-day, and 90-day late classes. Final class probabilities '
            'are computed via the chain rule: '
            'P(class = k, k > 0) = P(late) x P(class = k | late).'
            '\n\n'
            'ARCHITECTURAL RESTRICTION ON ORDINAL AND TWO-STAGE MODELS. '
            'The ordinal (8.2) and two-stage (8.3) classifiers are restricted '
            'to tree-based estimators (XGBoost, Random Forest, AdaBoost) because '
            'the training pipeline applies feature selection via mean decrease in '
            'impurity (MDI) using scikit-learn\'s SelectFromModel. This requires '
            'each estimator to expose a feature_importances_ attribute, which is '
            'only available for tree-based models. K-Nearest Neighbours and '
            'Gaussian Naive Bayes do not expose this attribute and are therefore '
            'excluded from the ordinal and two-stage experimental conditions.'
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Noted-only items
# ─────────────────────────────────────────────────────────────────────────────

NOTED_ITEMS = [
    {
        "id": "N1",
        "section": "3.4 — Modelling Component Narrative",
        "issue": (
            "The narrative for processes 6.0–12.0 does not mention Cox Survival "
            "Analysis Tuning (7.5) as a dedicated step or explain the dual "
            "baseline/enhanced training paths."
        ),
        "action": (
            "Update the introductory paragraph of Section 3.4 to state that the "
            "modelling component contains two concurrent pipelines: one baseline "
            "pipeline trained on the original feature set, and one enhanced "
            "pipeline trained on features augmented by survival analysis outputs. "
            "Reference processes 7.5 and 8.5 explicitly."
        ),
    },
    {
        "id": "N2",
        "section": "3.3 — Processing Component Narrative",
        "issue": (
            "The narrative for Process 5.0 (Data Cleaning) does not describe the "
            "data stream split into ML training data and survival analysis data."
        ),
        "action": (
            "The replacement text provided under D3 above addresses this. "
            "When implementing docx edits, apply the D3 narrative to the relevant "
            "paragraph in Section 3.3."
        ),
    },
    {
        "id": "N3",
        "section": "3.5 — Analysis Component DFD",
        "issue": (
            "Step 5 finalisation (re-train on the full dataset and save "
            "finalized_*.pkl model files) is not reflected in the Analysis DFD "
            "or its narrative."
        ),
        "action": (
            "Add a Process 17.0 'Finalise and export models' node to the "
            "Analysis DFD. Update Section 3.5 narrative to describe re-training "
            "on the combined train+test set and persisting the models."
        ),
    },
    {
        "id": "N4",
        "section": "3.2 — Conceptual Framework",
        "issue": (
            "The high-level Conceptual Framework diagram shows a 'Modelling' "
            "box but does not reflect Cox Survival Analysis as an internal "
            "sub-component."
        ),
        "action": (
            "Revise the Conceptual Framework diagram to split the Modelling box "
            "into two visible sub-tracks: 'Baseline ML pipeline' and "
            "'Survival-augmented ML pipeline', connected via the Cox tuning and "
            "feature generation nodes."
        ),
    },
    {
        "id": "N5",
        "section": "3.4 — Individual Model Descriptions",
        "issue": (
            "Descriptions for individual models (DT, RF, XGB, Ada, KNN, NB, "
            "Ordinal, Two-Stage) may not match the final implemented "
            "hyperparameter configurations stored in settings.json."
        ),
        "action": (
            "Cross-reference the hyperparameter grids in settings.json with the "
            "Chapter 3 model description tables and update any mismatches. Pay "
            "particular attention to n_estimators, max_depth, and learning_rate "
            "values for tree-based models."
        ),
    },
    {
        "id": "S_analysis",
        "section": "3.5 — Analysis Component DFD",
        "issue": (
            "The Analysis DFD (processes 13.0–16.0) has not been audited against "
            "the codebase in this revision cycle."
        ),
        "action": (
            "Perform a targeted trace of the analysis/reporting code to verify "
            "that classify_transactions (13.0), generate_summaries (14.0), "
            "payment_analysis (15.0), and visualization (16.0) match the "
            "implemented logic. Update the DFD if discrepancies are found."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Level-2 DFD inventory
# ─────────────────────────────────────────────────────────────────────────────

LEVEL2_DFDS = [
    {
        "file": "Level-2 DFD - 1.0 data importation.drawio",
        "parent": "1.0 Data Importation (Processing DFD)",
        "processes": ["1.1 Load Excel files (async, parallel)",
                      "1.2 Data type conversion",
                      "1.3 Datetime parsing",
                      "1.4 Lookup-based due date updates"],
    },
    {
        "file": "Level-2 DFD - 5.0 data cleaning.drawio",
        "parent": "5.0 Data Cleaning (Processing DFD)",
        "processes": ["5.1 Invoice building (InvoiceBuilder)",
                      "5.2 Feature engineering (FeatureEngineer)",
                      "5.3 Post-processing (InvoicePostProcessor)",
                      "5.4 Data stream split"],
    },
    {
        "file": "Level-2 DFD - 7.5 cox survival analysis tuning.drawio",
        "parent": "7.5 Cox Survival Analysis Tuning (Modelling DFD) [NEW]",
        "processes": ["7.5.1 Initialise hyperparameter grid (6 alpha x 3 l1_ratio)",
                      "7.5.2 K-Fold cross-validation",
                      "7.5.3 Fit CoxnetSurvivalAnalysis per fold",
                      "7.5.4 Score C-index (Harrell concordance)",
                      "7.5.5 Select best (alpha, l1_ratio)",
                      "7.5.6 Derive 9 optimal time points"],
    },
    {
        "file": "Level-2 DFD - 8.0 model building.drawio",
        "parent": "8.0 Model Building (Modelling DFD)",
        "processes": ["8.1 Base classifier training (AdaBoost, RF, XGB, DT, GNB, KNN)",
                      "8.2 Ordinal classifier training (Frank & Hall K-1 decomposition)",
                      "8.3 Two-stage ensemble training (6 combinations)"],
    },
    {
        "file": "Level-2 DFD - 8.5 survival feature generation.drawio",
        "parent": "8.5 Survival Feature Generation (Modelling DFD) [NEW]",
        "processes": ["8.5.1 Fit CoxnetSurvivalAnalysis (training data only)",
                      "8.5.2 Compute survival probability S(t) at 9 time points",
                      "8.5.3 Compute cumulative hazard H(t) at 9 time points",
                      "8.5.4 Compute risk score and expected survival time E[T]",
                      "8.5.5 Concatenate with original features -> Enhanced dataset"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Report generator
# ─────────────────────────────────────────────────────────────────────────────

class RevisionPlanner:

    def generate(self) -> str:
        lines = []
        today = date.today().isoformat()

        lines += [
            "# Thesis Chapter 3 — DFD Revision Plan",
            f"_Generated: {today}_",
            "",
            "---",
            "",
        ]

        # ── PART A: Implemented DFD changes ──────────────────────────────────
        lines += [
            "## Part A — Implemented Level-1 DFD Changes",
            "",
            "The following discrepancies have been **resolved** by editing the",
            "Level-1 DFD `.drawio` files in `docs/thesis_paper/diagrams/`.",
            "",
            "| ID | File | Change |",
            "|----|------|--------|",
            "| D1 | Level-1 DFD - modelling component.drawio | Added process 7.5 Cox Survival Analysis Tuning between 7.0 and 8.0; added edges: df_data_surv -> 7.5, Tuning parameters -> 8.0 |",
            "| D2 | Level-1 DFD - modelling component.drawio | Added process 8.5 Survival Feature Generation after 7.5; added edges: Tuned Cox model -> 8.5, Enhanced features -> 8.0 |",
            "| D3 | Level-1 DFD - processing component.drawio | Renamed single 'Cleaned data' edges from 5.0 to 'ML training data' and 'Survival analysis data' |",
            "| D4 | Level-1 DFD - modelling component.drawio | Added 'balancing strategy' annotation to the hyperparameter edge feeding Process 6.0 |",
            "",
        ]

        # ── PART B: Level-2 sub-DFDs generated ───────────────────────────────
        lines += [
            "## Part B — Generated Level-2 Sub-DFD Files",
            "",
            "New Level-2 DFD drawio files have been generated in `docs/thesis_paper/diagrams/`.",
            "",
        ]
        for dfd in LEVEL2_DFDS:
            lines += [
                f"### `{dfd['file']}`",
                f"**Parent process**: {dfd['parent']}",
                "",
                "| Sub-Process |",
                "|------------|",
            ]
            for p in dfd["processes"]:
                lines.append(f"| {p} |")
            lines.append("")

        # ── PART C: Narrative replacements ───────────────────────────────────
        lines += [
            "## Part C — Narrative Replacement Text",
            "",
            "The following paragraph replacements should be applied to",
            "`Beley-Reyes_Thesis2-ACM.docx` when the docx revision pass is performed.",
            "Each entry gives the **Section**, the **original text** (as a",
            "search anchor), and the **replacement text** to substitute.",
            "",
        ]
        for key, item in NARRATIVES.items():
            lines += [
                f"### {item['section']}",
                f"**{item['heading']}**",
                "",
                "**Original (search anchor):**",
                f"> {item['original']}",
                "",
                "**Replacement:**",
                "",
            ]
            for para in item["replacement"].split("\n\n"):
                lines.append(para.strip())
                lines.append("")

        # ── PART D: Noted-only items ──────────────────────────────────────────
        lines += [
            "## Part D — Noted-Only Discrepancies (Not Yet Implemented)",
            "",
            "The following discrepancies were identified but are deferred to a",
            "separate revision pass.",
            "",
        ]
        for item in NOTED_ITEMS:
            lines += [
                f"### {item['id']} — {item['section']}",
                "",
                f"**Issue**: {item['issue']}",
                "",
                f"**Recommended action**: {item['action']}",
                "",
            ]

        return "\n".join(lines)

    def write(self, path: Path) -> None:
        content = self.generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"Revision plan written -> {path}")
        print(f"  Lines: {len(content.splitlines())}")


if __name__ == "__main__":
    planner = RevisionPlanner()
    planner.write(OUTPUT_PATH)
