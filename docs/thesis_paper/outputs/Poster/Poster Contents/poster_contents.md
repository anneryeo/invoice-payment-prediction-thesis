# Poster Highlights — Thesis Key Points

---

## HEADER

**Title:**
SOLVING THE INVOICE PAYMENT PREDICTION PROBLEM (IPPP): A MULTI-STAGE APPROACH

**Researchers:**
Rafael Joseph T. Beley · Christine Julliane L. Reyes

**Department / School:**
School of Information Technology, Mapúa University

**Degree Program:**
Bachelor of Science in Data Science

**Date:**
June 2026

---

## INTRODUCTION

Unpaid tuition fees are a growing financial risk for Philippine private schools. Republic Act 11984 (No Permit, No Exam Prohibition Act) bars schools from withholding examination privileges for outstanding balances, eliminating the most direct enforcement mechanism and leaving institutions dependent on proactive cash-flow management.

At the partner institution (a private school in Montalban, Rizal), **Bad Debts Expense rose from PHP 244,111 (2019) to PHP 469,102 (2025)**, with 27 students in arrears and Days Sales Outstanding (DSO) reaching 34 days — despite flexible installment plans.

Machine learning has been widely applied to credit scoring, default prediction, and accounts-receivable optimization. However, existing invoice payment prediction studies share three critical gaps:
1. They use **aggregate student-level features**, discarding the line-item granularity at which invoices are actually issued.
2. They treat payment delay as a **nominal multi-class problem**, ignoring its inherent ordinal structure (on-time → 1–30 → 31–60 → 61+ days late).
3. They benchmark only **5–8 standard classifiers**, leaving ordinal decompositions, two-stage architectures, and survival-analysis features systematically untested.

---

## OBJECTIVES

This study develops and benchmarks a high-performance multi-stage machine learning pipeline for the educational Invoice Payment Prediction Problem (IPPP). Specifically:

1. Conduct a **large-scale benchmarking study** across 1,092 experimental configurations to identify the optimal model–preprocessing combination.
2. Develop a **two-stage ensemble architecture** that first separates delinquent from on-time invoices, then classifies delay severity.
3. Evaluate the impact of **survival-analysis-derived feature engineering** (Cox Proportional Hazards model) on classifier sensitivity and recall for late-payment brackets.
4. Provide an **empirically grounded framework** for private schools to transition from reactive collection practices to proactive receivables management.

---

## METHODOLOGY

The pipeline follows seven sequential stages illustrated in the flowchart below (`methodology_flowchart.svg`). Each stage corresponds to a concrete implementation component in the project codebase.

---

**① Data Acquisition.** Three Excel exports from the school's ERP system — a revenues file (itemized receivables with due and payment dates), an enrollees file (installment plan per school year), and a chart of accounts (category-to-SBU mapping) — form the raw inputs. The 11,440 records span academic years 2019–2025, with payment activity observed through March 31, 2026.

**② Pre-Processing.** Student identifiers are irreversibly hashed (pseudonymized) before any analysis. The three source files are merged into a line-item Credit Sales Fact Table. Days to Payment (DTP) is computed per invoice as the gap between due date and full-settlement date, then discretized into four ordinal brackets:

| Class | Label | DTP Range |
|-------|-------|-----------|
| 0 | On-Time | ≤ 0 days |
| 1 | Slightly Late | 1–30 days |
| 2 | Moderately Late | 31–60 days |
| 3 | Severely Late | 61+ days |

A temporal split at March 7, 2025 separates training (6,527 labeled records) from test data; invoices unpaid at the observation cutoff are treated as censored in Stage 4.

**③ Feature Engineering (Baseline — 40 features).** Forty variables across six groups are derived from the fact table at category-level granularity — not aggregate student balances:

| Group | Features | Count |
|-------|----------|-------|
| Raw Financial | gross receivables, discounts, adjustments, net receivable | 4 |
| DTP Historical | last 4 DTP values, weighted avg, trend, rolling std, max | 11 |
| Financial / Cumulative | cumsum due/paid, opening balance flag, payment ratio | 5 |
| Behavioral | previous bracket, early payer flag, on-time streak | 3 |
| Payment Plan | one-hot plan type (A–E + NaN), ordinal risk score | 13 |
| Temporal | due month, due quarter | 4 |

**④ Survival Analysis (Enhanced — +5 features).** A Cox Proportional Hazards model (`lifelines`) is fitted on the censored training subset to model time-to-payment. Five derived features extend the Baseline into an Enhanced feature regime: `partial_hazard`, `log_partial_hazard`, `expected_survival_time`, `survival_probability`, and `cumulative_hazard`. Both regimes are benchmarked to isolate the marginal utility of survival features.

**⑤ Class Balancing (7 strategies).** With ~76% of invoices on-time (Class 0), seven resampling strategies are applied to the training partition: no resampling, SMOTE, Borderline-SMOTE, SMOTE+Tomek, and three hybrid undersampling ratios (@0.5, @0.7, @0.9). Each strategy × model combination is an independent experiment.

**⑥ Model Training & Benchmarking (1,092 configurations).** Fifteen architectures across three families are trained under every strategy–regime combination via grid search (~10 hyperparameter configs per model):

| Family | Count | Architectures |
|--------|-------|--------------|
| Base Classifiers | 6 | AdaBoost, Random Forest, XGBoost, Decision Tree, GNB, KNN |
| Two-Stage Ensembles | 6 | XGB→Ada, RF→Ada, RF→RF, XGB→RF, XGB→XGB, Ada→XGB |
| Ordinal Classifiers | 3 | Ordinal AdaBoost, Ordinal RF, Ordinal XGBoost (Frank–Hall decomposition) |

Two-stage pipelines run a binary Stage 1 (on-time vs. late) then pass predicted-late invoices to a multi-class Stage 2 (severity: Class 1/2/3).

**⑦ Evaluation & Deployment.** All 1,092 configurations are ranked by **macro-averaged F1-score** (primary, corrects for 76% Class 0 dominance) and ROC-AUC (secondary). Spearman ρ = 0.732 (p < 0.001) between the two metrics validates F1 as a reliable selector. The best configuration is operationalized in a **Dash (Python/Plotly) web prototype** with five functional screens: Setup Wizard, KPI Dashboard, Invoice Prediction Drilldown, Comparative Model Dashboard, and Audit Logs.

---

## RESULTS

### RQ1: Two-Stage vs. Ordinal vs. Base Classifiers
| Model Family | Peak F1 | Peak AUC | Best Config |
|---|---|---|---|
| **Two-Stage** | **0.6003** | **0.8919** | XGB→Ada + Borderline-SMOTE |
| Ordinal | 0.5621 | 0.8205 | Ordinal AdaBoost + Hybrid@0.5 |
| Base | 0.5589 | 0.8607 | XGBoost + Hybrid@0.5 |

The two-stage XGBoost→AdaBoost pipeline **outperformed the best ordinal model by +0.038 F1** and the best base classifier by **+0.041 F1**. The second-best ensemble, TS RF→Ada, reached F1 = 0.588 under Borderline-SMOTE. Gains are not universal across the two-stage family: pipelines that route Stage 2 to XGBoost (TS XGB→XGB, TS Ada→XGB) underperform their corresponding base classifiers, indicating that hierarchical decomposition is most effective when paired with a **complementary, lower-variance learner** (AdaBoost) in Stage 2 rather than a second high-capacity booster.

### RQ2: Survival Feature Impact
- **Positive lift:** KNN (+0.0061 F1) and Gaussian Naive Bayes (+0.0050 F1) — simpler models benefit from pre-computed temporal risk statistics.
- **Negligible to negative:** Random Forest, XGBoost, AdaBoost — high-capacity tree ensembles independently approximate survival patterns, making Cox features redundant or noisy.

**Finding:** Survival feature engineering has **conditional, not universal** utility — allocate effort based on the downstream learner's capacity.

### RQ3: Class-Balancing Strategy
- **Borderline-SMOTE** consistently outperformed vanilla SMOTE and configurations without resampling across all model families.
- Resampling is **essential for two-stage models**: TS XGB→Ada scores 0.538 with no resampling vs. **0.600 with Borderline-SMOTE** — a +0.062 swing. All ensemble "None" configurations fall in the 0.516–0.541 range.
- Peak F1 difference between Borderline-SMOTE and vanilla SMOTE: **≈ 0.022** for top ensembles (TS XGB→Ada: 0.600 vs. 0.578).
- **Hybrid undersampling** is a stable alternative: TS XGB→Ada at Hybrid@0.5 = 0.583, only 0.017 below Borderline-SMOTE.
- Weak classifiers swing most dramatically: GNB rises from 0.461 (None) to 0.502 (Hybrid@0.5); KNN similarly volatile.
- Spearman ρ = 0.732 (p < 0.001) between F1 and AUC confirms resampling drives **threshold placement**, not underlying class separability — AUC spread across strategies is narrower than F1 spread.
- Strong tree ensembles (RF, XGBoost, AdaBoost) are relatively robust to strategy choice; weaker learners (KNN, GNB) are highly sensitive — strategy selection should be guided by the downstream learner's capacity.

### RQ4: Feature Importance (Granular vs. Aggregate)
The three most predictive features across all model families:
1. `prev_bracket` — the student's most recent payment bracket
2. `dtp_wavg` — recency-weighted average of historical days-to-payment
3. `opening_balance_flag` — whether the student carried an unpaid balance into the period

All three are **behavioral-historical features only available at line-item granularity**, confirming the study's core design rationale. LDA corroborates: LD1 explains 79.5% of between-class separation, and 95.9% of separation among delinquent classes alone.

### Prototype System
A **Dash (Python/Plotly)** web application operationalizes the best model through 5 screens:
- **Initial Setup Wizard** — upload source files, trigger training pipeline
- **KPI Dashboard** — live F1/AUC metrics, payment bracket distribution, top model ranking
- **Invoice Prediction Drilldown** — paginated predictions with confidence scores, filterable by bracket, exportable to CSV
- **Comparative Model Dashboard** — ROC curves, confusion matrices, F1/AUC charts across all 1,092 configurations
- **Audit Logs** — timestamped record of every prediction run and settings change

---

## CONCLUSION

Across 1,092 controlled configurations, four structural findings emerge:

1. **Hierarchical architectures dominate.** The two-stage XGBoost→AdaBoost pipeline sets the performance ceiling (F1 = 0.6003, AUC = 0.8919), outperforming single-stage and ordinal approaches.
2. **Resampling is indispensable.** With ~76% of invoices on-time, Borderline-SMOTE is essential to achieve meaningful recall for the late-payment brackets that carry the greatest financial risk.
3. **Ordinal structure yields modest but consistent gains.** Exploiting the ordered nature of delay brackets improves performance, but less so than multi-stage decomposition.
4. **Survival features are model-specific.** They benefit probabilistic and distance-based learners; they offer redundant signal to high-capacity tree ensembles.

A macro-F1 of 0.60 on a four-class imbalanced problem means the model correctly brackets ~60 of every 100 eventually-late invoices — and most remaining errors land in adjacent brackets. Against the status quo of zero forward visibility, this enables finance officers to **concentrate outreach on the highest-risk invoices before due dates lapse**: earlier reminders, proactive installment restructuring, and social welfare referrals — in place of after-the-fact collection that RA 11984 has rendered largely unenforceable.

The framework is directly transferable to any Philippine private school sharing the same installment-billing structure; the benchmarking protocol is fully replicable; and the open-sourced prototype operationalizes the winning configuration. Predictive receivables management, in aligning financial sustainability with educational equity, becomes an instrument of responsible institutional governance.

---

## ACKNOWLEDGEMENTS


The researchers would like to express their heartfelt gratitude to their thesis adviser Mr. Joel C. De Goma and the panel members for their guidance, invaluable feedback, and constructive evaluations throughout the study. Appreciation is also extended to the partner institution for granting access to the institutional dataset and for their continued support.

Most importantly, the researchers wish to acknowledge their families and loved ones, whose unwavering encouragement, patience, and understanding provided strength and inspiration during the entire research journey. Their sacrifices, moral support, and constant belief in the researchers’ abilities made the completion of this study possible.

---

*Source: Beley, R. J. T. & Reyes, C. J. L. (2026). Solving the Invoice Payment Prediction Problem (IPPP): A Multi-Stage Approach. Bachelor of Science in Data Science Thesis, School of Information Technology, Mapúa University.*
