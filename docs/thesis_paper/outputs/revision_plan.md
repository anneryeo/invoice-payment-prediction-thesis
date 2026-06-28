# Thesis Chapter 3 — DFD Revision Plan
_Generated: 2026-06-28_

---

## Part A — Implemented Level-1 DFD Changes

The following discrepancies have been **resolved** by editing the
Level-1 DFD `.drawio` files in `docs/thesis_paper/diagrams/`.

| ID | File | Change |
|----|------|--------|
| D1 | Level-1 DFD - modelling component.drawio | Added process 7.5 Cox Survival Analysis Tuning between 7.0 and 8.0; added edges: df_data_surv -> 7.5, Tuning parameters -> 8.0 |
| D2 | Level-1 DFD - modelling component.drawio | Added process 8.5 Survival Feature Generation after 7.5; added edges: Tuned Cox model -> 8.5, Enhanced features -> 8.0 |
| D3 | Level-1 DFD - processing component.drawio | Renamed single 'Cleaned data' edges from 5.0 to 'ML training data' and 'Survival analysis data' |
| D4 | Level-1 DFD - modelling component.drawio | Added 'balancing strategy' annotation to the hyperparameter edge feeding Process 6.0 |

## Part B — Generated Level-2 Sub-DFD Files

New Level-2 DFD drawio files have been generated in `docs/thesis_paper/diagrams/`.

### `Level-2 DFD - 1.0 data importation.drawio`
**Parent process**: 1.0 Data Importation (Processing DFD)

| Sub-Process |
|------------|
| 1.1 Load Excel files (async, parallel) |
| 1.2 Data type conversion |
| 1.3 Datetime parsing |
| 1.4 Lookup-based due date updates |

### `Level-2 DFD - 5.0 data cleaning.drawio`
**Parent process**: 5.0 Data Cleaning (Processing DFD)

| Sub-Process |
|------------|
| 5.1 Invoice building (InvoiceBuilder) |
| 5.2 Feature engineering (FeatureEngineer) |
| 5.3 Post-processing (InvoicePostProcessor) |
| 5.4 Data stream split |

### `Level-2 DFD - 7.5 cox survival analysis tuning.drawio`
**Parent process**: 7.5 Cox Survival Analysis Tuning (Modelling DFD) [NEW]

| Sub-Process |
|------------|
| 7.5.1 Initialise hyperparameter grid (6 alpha x 3 l1_ratio) |
| 7.5.2 K-Fold cross-validation |
| 7.5.3 Fit CoxnetSurvivalAnalysis per fold |
| 7.5.4 Score C-index (Harrell concordance) |
| 7.5.5 Select best (alpha, l1_ratio) |
| 7.5.6 Derive 9 optimal time points |

### `Level-2 DFD - 8.0 model building.drawio`
**Parent process**: 8.0 Model Building (Modelling DFD)

| Sub-Process |
|------------|
| 8.1 Base classifier training (AdaBoost, RF, XGB, DT, GNB, KNN) |
| 8.2 Ordinal classifier training (Frank & Hall K-1 decomposition) |
| 8.3 Two-stage ensemble training (6 combinations) |

### `Level-2 DFD - 8.5 survival feature generation.drawio`
**Parent process**: 8.5 Survival Feature Generation (Modelling DFD) [NEW]

| Sub-Process |
|------------|
| 8.5.1 Fit CoxnetSurvivalAnalysis (training data only) |
| 8.5.2 Compute survival probability S(t) at 9 time points |
| 8.5.3 Compute cumulative hazard H(t) at 9 time points |
| 8.5.4 Compute risk score and expected survival time E[T] |
| 8.5.5 Concatenate with original features -> Enhanced dataset |

## Part C — Narrative Replacement Text

The following paragraph replacements should be applied to
`Beley-Reyes_Thesis2-ACM.docx` when the docx revision pass is performed.
Each entry gives the **Section**, the **original text** (as a
search anchor), and the **replacement text** to substitute.

### 3.3.2 — Data Cleaning (Processing Component DFD)
**Process 5.0 Output Data Flows**

**Original (search anchor):**
> The cleaned data output from Process 5.0 is passed to the modelling component for further preparation.

**Replacement:**

After cleaning, the pipeline produces two distinct data streams. The first stream — referred to as ML training data — contains only records where the censor indicator equals one (i.e., the event was observed). Survival-specific columns are dropped from this stream before it is forwarded to the modelling component for balancing, partitioning, and classifier training. The second stream — referred to as survival analysis data — retains all records and all survival columns (time-to-event T and event indicator E). This stream is forwarded exclusively to the Cox Proportional Hazards tuning sub-process (7.5) and subsequently to the survival feature generation sub-process (8.5).

### 3.4.x — Cox Survival Analysis Tuning (Modelling Component DFD)
**Process 7.5 — Cox Survival Analysis Tuning [NEW]**

**Original (search anchor):**
> (No corresponding narrative existed; this process was absent from the Chapter 3 DFD and manuscript.)

**Replacement:**

Process 7.5 performs hyperparameter tuning for the Cox Proportional Hazards model (CoxnetSurvivalAnalysis, scikit-survival) using the survival analysis data stream produced by Process 5.0. A grid of 18 hyperparameter combinations is evaluated: six values of the regularisation strength alpha and three values of the elastic-net mixing parameter l1_ratio. For each combination, k-fold cross-validation is applied, and each fold is scored using Harrell's concordance index (C-index). The combination that maximises the mean C-index across folds is selected as the best set of hyperparameters. Following hyperparameter selection, nine optimal time points are derived from the observed event distribution using a slope-change algorithm applied to the Kaplan-Meier survival curve. These nine time points are stored alongside the best hyperparameters and are used by Process 8.5 to generate per-sample survival features.

### 3.4.x — Survival Feature Generation (Modelling Component DFD)
**Process 8.5 — Survival Feature Generation [NEW]**

**Original (search anchor):**
> (No corresponding narrative existed; this process was absent from the Chapter 3 DFD and manuscript.)

**Replacement:**

Process 8.5 generates survival-derived features that augment the original feature set before classifier training. Using the best hyperparameters obtained from Process 7.5, a CoxnetSurvivalAnalysis model is fitted on the training split of the survival analysis data. The fitted model is then applied to both training and test splits to compute, for each sample, the following features at each of the nine optimal time points: the survival probability S(t) and the cumulative hazard H(t). Additionally, a scalar risk score and the expected survival time E[T] are computed for each sample. These survival-derived features are concatenated with the original feature matrix to produce an enhanced dataset. Both the original and enhanced datasets are forwarded to Process 8.0 (Model Building), enabling a paired comparison between baseline and survival-augmented classification performance.

### 3.4.1 — Data Preparation (Modelling Component DFD)
**Process 6.0 — Data Preparation (balancing annotation)**

**Original (search anchor):**
> The data preparation process encodes categorical labels and normalises numerical features before model training.

**Replacement:**

Process 6.0 performs four sequential operations on the ML training data stream received from Process 5.0. First, ordinal labels are encoded: on-time, 30-day, 60-day, and 90-day payment outcomes are mapped to integer classes 0–3. Second, a temporal train/test split is applied by sorting records on the instalment due date and assigning the earliest 80 percent of records to the training partition and the latest 20 percent to the test partition, preventing data leakage from future records. Third, a user-selected balancing strategy is applied to the training partition only. Five strategies are supported: SMOTE, Borderline SMOTE, SMOTEENN, SMOTETomek, and the custom HybridBalance strategy; each strategy is treated as a separate experimental condition. Fourth, min-max normalisation is applied to all numerical features. An optional linear discriminant analysis (LDA) projection may also be applied at this stage to reduce dimensionality before forwarding the prepared data to Process 7.0.

### 3.4.4 — Model Building (Modelling Component DFD)
**Process 8.0 — Model Building [EXPANDED]**

**Original (search anchor):**
> The model building process trains the selected classifiers on the prepared training data.

**Replacement:**

Process 8.0 trains fifteen classifiers organised into three sub-categories: six base classifiers, three ordinal classifiers, and six two-stage ensemble classifiers.

BASE CLASSIFIERS (8.1). Six classifiers are trained independently on both the original and survival-enhanced feature sets: AdaBoost (AdaBoostClassifier), Random Forest (RandomForestClassifier), XGBoost (XGBClassifier), Decision Tree (DecisionTreeClassifier), Gaussian Naive Bayes (GaussianNB), and K-Nearest Neighbours (KNeighborsClassifier). Each classifier is trained under each of the five balancing strategies, yielding a full experimental matrix.

ORDINAL CLASSIFIERS (8.2). Three ordinal classifiers are built using the Frank and Hall (2001) binary decomposition method: Ordinal AdaBoost, Ordinal Random Forest, and Ordinal XGBoost. For four payment outcome classes (0–3), the method trains K-1 = 3 binary classifiers: Classifier 0 learns P(y > 0) — on-time vs any late; Classifier 1 learns P(y > 1) — at most 30-day late vs 60-day or more late; Classifier 2 learns P(y > 2) — at most 60-day late vs 90-day late. Class probabilities are recovered as differences: P(class = k) = P(y > k-1) - P(y > k), with boundary values P(y > -1) = 1 and P(y > K-1) = 0. Monotonicity is enforced by clipping negative differences to zero and re-normalising the resulting distribution.

TWO-STAGE ENSEMBLE CLASSIFIERS (8.3). Six two-stage ensemble classifiers are trained, each combining two tree-based estimators: XGBoost-XGBoost, XGBoost-Random Forest, Random Forest-Random Forest, XGBoost-AdaBoost, Random Forest-AdaBoost, and AdaBoost-XGBoost. Stage 1 is a binary classifier trained on the full dataset to predict P(late) — the probability that a payment is late (any class other than on-time). Stage 2 is a multiclass classifier trained only on the late subset, predicting P(class = k | late) for the 30-day, 60-day, and 90-day late classes. Final class probabilities are computed via the chain rule: P(class = k, k > 0) = P(late) x P(class = k | late).

ARCHITECTURAL RESTRICTION ON ORDINAL AND TWO-STAGE MODELS. The ordinal (8.2) and two-stage (8.3) classifiers are restricted to tree-based estimators (XGBoost, Random Forest, AdaBoost) because the training pipeline applies feature selection via mean decrease in impurity (MDI) using scikit-learn's SelectFromModel. This requires each estimator to expose a feature_importances_ attribute, which is only available for tree-based models. K-Nearest Neighbours and Gaussian Naive Bayes do not expose this attribute and are therefore excluded from the ordinal and two-stage experimental conditions.

## Part E — Post-Figure Narratives for Level-2 DFDs

_Implemented: 2026-06-29_

The last docx revision pass (commit `3510afa`) inserted introductory sentences **before** each Level-2 DFD figure but did not add explanatory prose **after** the figure caption. This part adds a narrative paragraph immediately following each of the five Level-2 DFD captions, explaining what each sub-process in the diagram does. Content is grounded in `code_pipeline_trace.md`.

| DFD | Caption anchor keyword | Narrative added |
|-----|------------------------|-----------------|
| Level-2 DFD 1.0 Data Importation | "1.0 data importation" | Sub-processes 1.1–1.4: async load, type conversion, datetime parsing, due-date lookup update |
| Level-2 DFD 5.0 Data Cleaning | "5.0 data cleaning" | Sub-processes 5.1–5.4: InvoiceBuilder, FeatureEngineer, PostProcessor, data stream split |
| Level-2 DFD 7.5 Cox Survival Analysis Tuning | "7.5 cox survival" | Sub-processes 7.5.1–7.5.6: grid init, k-fold CV, model fit, C-index scoring, best selection, 9 time points |
| Level-2 DFD 8.5 Survival Feature Generation | "8.5 survival feature" | Sub-processes 8.5.1–8.5.5: Cox fit, S(t), H(t), risk score/E[T], enhanced dataset concat |
| Level-2 DFD 8.0 Model Building | "8.0 model building" | Sub-processes 8.1–8.3: base classifiers, ordinal (Frank & Hall), two-stage ensembles |

`chapter3_methodology.md` was also updated to reflect these new sections as a reference document.

---

## Part D — Noted-Only Discrepancies

_N1 and N2 implemented: 2026-06-29_

### N1 — 3.4 — Modelling Component Narrative ✓ Implemented

**Issue**: The narrative for processes 6.0–12.0 did not mention Cox Survival Analysis Tuning (7.5) as a dedicated step or explain the dual baseline/enhanced training paths.

**Resolution**: The paragraph immediately following Figure 3.3 caption was replaced to describe the two concurrent training pipelines (baseline and survival-augmented), with explicit references to Process 7.5 and Process 8.5.

### N2 — 3.3 — Processing Component Narrative ✓ Implemented

**Issue**: The narrative for the Figure 3.2 (Processing Component Level-1 DFD) did not describe the data stream split into ML training data and survival analysis data.

**Resolution**: The paragraph immediately following the Figure 3.2 caption was updated to state that Process 5.0 writes records to the modeling cache as two distinct data streams — ML training data and survival analysis data — forwarded to separate downstream processes.
