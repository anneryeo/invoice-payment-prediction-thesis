# Discrepancy Analysis: DFD vs Code Pipeline

_Compares Level-1 DFD elements (Phase 2.1) against the actual code pipeline (Phase 2.2)._

## ✅ In Sync

| ID | DFD | Process | Priority | Description |
|----|-----|---------|----------|-------------|
| S1 | Processing | 1.0 | LOW | Data importation maps to base64 Excel decode + Revenues async loader |
| S2 | Processing | 2.0 | LOW | Time to payment maps to DTP lag feature computation |
| S3 | Processing | 3.0 | LOW | Student information maps to demographic feature engineering |
| S4 | Processing | 4.0 | LOW | Payment history maps to 30/60/90 day bracket allocation |
| S5 | Processing | 5.0 | LOW | Data cleaning maps to CreditSalesProcessor orchestration |
| S6 | Modelling | 6.0 | LOW | Data preparation maps to DataPreparer (encode → split → resample → normalize) |
| S7 | Modelling | 7.0 | LOW | Data partitioning maps to temporal train/test split |
| S8 | Modelling | 8.0 | LOW | Model building maps to SurvivalExperimentRunner parallel training |
| S9 | Modelling | 9.0–12.0 | LOW | Feature selection, model execution, evaluation, combination all present |
| S10 | Analysis | 13.0–16.0 | LOW | Classify, summarize, payment analysis, visualization all present in Step 4/5 |

## 🔴 Missing DFD Processes (IMPLEMENT)

| ID | DFD | Process | Priority | Description |
|----|-----|---------|----------|-------------|
| D1 | Modelling | 7.5 (new) | HIGH | Cox Survival Analysis Tuning is a major standalone phase with no DFD process |
| D2 | Modelling | 8.5 (new) | HIGH | Survival Feature Generation creates a separate 'enhanced' dataset not shown in DFD |

## 🟠 Missing Data Flows (IMPLEMENT)

| ID | DFD | Process | Priority | Description |
|----|-----|---------|----------|-------------|
| D3 | Processing | 5.0 | HIGH | After Data cleaning (5.0), code produces two distinct output streams; DFD shows only one |
| D4 | Modelling | 6.0 | MEDIUM | Data preparation handles SMOTE balancing but DFD does not label it as such |
| D5 | Modelling | 8.0 | HIGH | Model building receives both baseline and enhanced feature sets; DFD shows only one data flow in |

## 🟡 Mislabeled / Underspecified (IMPLEMENT)

| ID | DFD | Process | Priority | Description |
|----|-----|---------|----------|-------------|
| D0 | Modelling | 8.0 | HIGH | Process 8.0 'Model building' label understates the 15-variant architecture inside it |

## 📝 Noted Only — Out of Scope for This Revision

| ID | DFD | Process | Priority | Description |
|----|-----|---------|----------|-------------|
| N0 | Modelling | 8.0 | HIGH | Chapter 3 Section 3.5 narrative on 8.0 needs to include the 15-variant model table |
| N1 | Modelling | 6.0–12.0 | MEDIUM | Modelling DFD narrative does not mention Cox tuning as a dedicated phase |
| N2 | Processing | 5.0 | MEDIUM | Section 3.2 narrative for 5.0 should describe the data stream split |
| N3 | Analysis | — | LOW | Step 5 model finalization (re-train on full dataset) not shown in Analysis DFD |
| N4 | All | — | LOW | Conceptual Framework diagram's Modelling box doesn't reflect Cox as a sub-component |
| N5 | All | — | LOW | Individual model hyperparameter configs in Chapter 3 may not match settings.json |

## Detailed Findings

### D1 — Cox Survival Analysis Tuning is a major standalone phase with no DFD process

**DFD**: Modelling  |  **Process**: 7.5 (new)  |  **Priority**: HIGH

**Code evidence**: tune_cox_model(): CoxHyperparameterTuner grid search 18 combos (6 alpha × 3 l1_ratio). K-Fold CV; scored by Harrell C-index. Outputs best_surv_parameters, best_time_points (9). This runs AFTER data partitioning and BEFORE model building — a dedicated phase.


### D2 — Survival Feature Generation creates a separate 'enhanced' dataset not shown in DFD

**DFD**: Modelling  |  **Process**: 8.5 (new)  |  **Priority**: HIGH

**Code evidence**: generate_survival_features(): fits CoxnetSurvivalAnalysis on training data. Computes S(t), H(t) at 9 time points, risk_score, E[T] per sample. Concatenates with original features → enhanced dataset distinct from baseline.


### D3 — After Data cleaning (5.0), code produces two distinct output streams; DFD shows only one

**DFD**: Processing  |  **Process**: 5.0  |  **Priority**: HIGH

**Code evidence**: clean_datasets() returns df_data (censor==1, survival cols dropped) AND df_data_surv (full, survival cols intact). The DFD shows a single 'Cleaned data' output. Both streams have distinct downstream consumers: baseline ML vs Cox tuning.


### D4 — Data preparation handles SMOTE balancing but DFD does not label it as such

**DFD**: Modelling  |  **Process**: 6.0  |  **Priority**: MEDIUM

**Code evidence**: DataPreparer.resample() runs SMOTE, BorderlineSMOTE, SMOTEENN, SMOTETomek, or HybridBalance as a required pre-training step. The 'Model hyperparameters' edge into 6.0 should include 'balancing strategy' as an explicit input label.


### D5 — Model building receives both baseline and enhanced feature sets; DFD shows only one data flow in

**DFD**: Modelling  |  **Process**: 8.0  |  **Priority**: HIGH

**Code evidence**: SurvivalExperimentRunner trains each model TWICE: once with baseline features (X_train) and once with survival-augmented features (X_surv_train). The DFD only shows a single 'Training data' input to Model building.


### D0 — Process 8.0 'Model building' label understates the 15-variant architecture inside it

**DFD**: Modelling  |  **Process**: 8.0  |  **Priority**: HIGH

**Code evidence**: 8.0 covers: 6 base classifiers (Ada, RF, XGB, DT, GNB, KNN), 3 ordinal classifiers (Ordinal-Ada/RF/XGB via Frank & Hall K-1 decomposition), 6 two-stage ensembles (XGB/RF/Ada × XGB/RF/Ada combinations). Sub-DFD 8.0 needed to show the three sub-categories.

