# Utilizing Machine Learning to Solve the Invoice Payment Prediction Problem (IPPP)

> **Undergraduate Thesis — Bachelor of Science in Data Science**
> R.J.T. Beley · C.J.L. Reyes · J. De Goma

---
## Description
> **Utilizing Machine Learning to Solve the Invoice Payment Prediction Problem (IPPP)**: 
> This undergraduate thesis develops a production-ready ML classification system that predicts how long invoices will remain unpaid. Using pseudonymized educational institution data, the system classifies invoices into four payment brackets (On-Time, 1–30 days late, 31–60 days late, 61+ days late) to support cash flow forecasting and accounts receivable management.
> The approach combines payment behavior analytics, Cox proportional hazards survival modeling, and a comprehensive comparison of 15 classifier architectures across 7 class-balancing strategies. All 1092 experiments are logged to an SQLite results database with full traceability, feature importance tracking, and cross-validation metrics. Results include an interactive Dash web dashboard for model inspection, invoice-level prediction, and audit logging.

## Overview

This repository contains the full research codebase for our thesis that develops a machine-learning pipeline to classify student invoices by their expected **payment bracket** — how long after the due date an invoice will be fully settled. The four target classes are:

| Class       | Definition                     |
| ----------- | ------------------------------ |
| `on_time` | Paid on or before the due date |
| `30_days` | Paid 1–30 days overdue        |
| `60_days` | Paid 31–60 days overdue       |
| `90_days` | Paid 61+ days overdue          |

Accurate bracket prediction enables an educational institution to:

- Forecast cash flow and accounts receivable
- Prioritize collection efforts
- Assess invoice-level repayment risk

The dataset covers student enrollee revenue records (through March 31, 2026) from a **pseudonymized** educational institution.

---

## Authors / Developers

| Name                      | Role                 |
| ------------------------- | -------------------- |
| **RJ Beley**        | Co-author, developer |
| **Christine Reyes** | Co-author, developer |
| **Joel De Goma**    | Research Advisor     |

---

## Database Privacy Notice

> **The source database is private and is NOT included in this repository.**

The `data/` folder references Excel exports that were manually downloaded from internal school systems. These files contain sensitive student financial records and must never be committed, pushed, or shared publicly. Raw database exports are excluded via `.gitignore`.

Anyone replicating this study must supply their own institutional data that can be transformed into the expected schema via the `CreditSalesProcessor` pipeline. See [FEATURE_REFERENCE.md](FEATURE_REFERENCE.md) for the full column reference, feature descriptions, and required data types.

---

## Methodology

### 1. Data & Feature Engineering

Raw data is sourced from three Excel files (pseudonymized and stored in `database/`):

- **Revenues** — itemized receivables, discounts, adjustments, payment dates
- **Enrollees** — student enrollment records per school year
- **Chart of Accounts** — account mapping and categorization

The `CreditSalesProcessor` class (`src/modules/feature_engineering/credit_sales_machine_learning.py`) merges these sources and produces the core credit sales dataset. Key engineered features include:

**Days-to-Payment (DTP) Features**

- `dtp_1` through `dtp_4` — days to full payment for the most recent 4 invoices
- `dtp_avg`, `dtp_wavg` — simple and weighted average DTP
- `dtp_2_trend`, `dtp_3_trend` — trend slopes across recent invoices
- `days_since_last_payment`, `dtp_rolling_std`, `dtp_max`

**Financial Features**

- `gross_receivables`, `amount_discounted`, `adjustments`, `credit_sale_amount`
- `amount_due_cumsum`, `amount_paid_cumsum`, `opening_balance`

**Behavioral / Historical Features**

- `prev_bracket` — prior payment bracket
- `payment_ratio` — cumulative paid-to-due ratio
- `early_payer_flag`, `on_time_streak`
- `opening_balance_flag`

**Temporal Features**

- `due_month`, `due_quarter`
- Consecutive school-year enrollment streak (`src/modules/feature_engineering/consecutive_years.py`)

**Payment Plan Features**

- One-hot encoded plan types (A / B / C / D / E / None)
- `plan_type_risk_score` — ordinal risk encoding per plan

### 2. Survival Analysis (Cox Proportional Hazards)

Before classification, a **Cox PH model** (via `lifelines`) is fitted on censored training rows to generate time-dependent survival features. The Cox model achieves a **C-index of 0.7817**.

Key safeguard: the Cox model is fitted exclusively on the training set and applied separately to train and test splits to prevent data leakage.

Survival features are added to form the **"enhanced"** feature phase (vs. the "baseline" phase without them).

### 3. Resampling / Class Balancing

Seven balance strategies are tested per model:

| Strategy             | Description                           |
| -------------------- | ------------------------------------- |
| `none`             | No resampling                         |
| `smote`            | Standard SMOTE                        |
| `borderline_smote` | Borderline-SMOTE                      |
| `smote_tomek`      | SMOTE + Tomek Links                   |
| `hybrid@0.5`       | Undersampling + SMOTE (threshold 0.5) |
| `hybrid@0.7`       | Undersampling + SMOTE (threshold 0.7) |
| `hybrid@0.9`       | Undersampling + SMOTE (threshold 0.9) |

### 4. Classification Models (15 Types)

**Base classifiers (6):**

- AdaBoost, Decision Tree, Gaussian Naive Bayes, K-Nearest Neighbor, Random Forest, XGBoost (GPU-accelerated)

**Ordinal wrappers (3):**

- Ordinal Classifier applied to XGBoost, Random Forest, AdaBoost — respects the natural ordering of the 4 payment brackets

**Two-stage stacking classifiers (6):**
First-stage binary classifier separates on-time vs. delinquent; second-stage classifies delinquency severity:

- XGB → Random Forest, XGB → AdaBoost
- RF → Random Forest, RF → AdaBoost
- Ada → XGBoost, Ada → Random Forest

### 5. Experiment Scale

```
1092 total experiments
= 15 model types
× 7 balance strategies
× ~10 hyperparameter sets per model
× 2 feature phases (baseline / enhanced)
```

All experiments are logged to a **SQLite results database** (`results/2026_04_18_02/results.db`) with tables for:

- `experiments` — one row per experiment (model, strategy, params, phase)
- `metrics` — train/test accuracy, F1, AUC, precision, recall per experiment
- `features` — feature importance per experiment
- `class_mappings` — label encodings
- `survival_results` — Cox model parameters and time points
- `metadata` — run timestamps and elapsed time

### 6. Parallel Execution

Experiments are run in parallel via `run_models_parallel.py` (Windows multiprocessing). XGBoost runs sequentially due to GPU constraints. Joblib uses a threading backend to avoid nested-parallelism issues on Windows.

The notebook is executed headlessly using **Papermill** via `run_experiment.py`.

### 7. Linear Discriminant Analysis (Exploratory)

LDA is used as an exploratory analytical tool (not for final classification) via `src/modules/exploratory_data_analysis/linear_discriminant_analysis.py`:

- 4-class LDA: LD1 explains **79.5%** of separation variance; top separating features are `opening_balance_flag`, `opening_balance (log1p)`, `prev_bracket`, `dtp_wavg`
- 3-class LDA (delinquent only, excluding on-time): LD1 explains **95.9%** of separation variance

### 8. Web Dashboard

A **Dash** (Plotly) web application (`run_app.py`) provides an interactive frontend:

- Multi-step setup wizard for uploading data and triggering the pipeline
- KPI dashboard with class distribution and payment timelines
- Per-model performance and feature importance views
- Invoice-level drilldown with payment projections
- Audit logs and export options

---

## Repository Structure

```
.
├── run_app.py                      # Python entry point for the Dash web app
├── run_experiment.py               # Headless notebook runner (Papermill)
├── settings.json                   # Experiment configuration
├── .gitignore
├── FEATURE_REFERENCE.md
│
├── database/                       # Source Excel datasets (pseudonymized)
│   ├── revenues_pseudonymized.xlsx
│   ├── enrollees_pseudonymized.xlsx
│   └── chart_of_accounts.xlsx
│
├── results/                        # SQLite results databases from experiment runs
│   └── 2026_04_18_02/              # ✓ Latest valid results (1092 experiments)
│       └── results.db
│
├── src/
│   ├── app/                        # Dash app components (screens, utils, assets)
│   ├── modules/                    # Core domain modules
│   │   ├── exploratory_data_analysis/
│   │   ├── feature_engineering/    # Data processing & feature engineering
│   │   └── machine_learning/       # ML models, pipelines, and training utils
│   └── utils/                      # Shared utility scripts (loaders, pseudonymizer)
│
├── notebooks/
│   ├── eda/                        # Exploratory data analysis notebooks
│   ├── results/                    # Results analysis & visualization notebooks
│   └── training/                   # Primary ML training notebooks
│
├── environments/                   # Conda/pip environment specs
├── data/                           # Generated results (graphics, results)
│   ├── eda_results/                # Outputs from EDA notebooks
│   └── Results_Base-Graphics-for-Paper/ # Figures used in the paper
└── tests/                          # Unit and integration tests
```

> **Note:** Root-level directories like `machine_learning/` or `app/` (without `src/` prefix) are remnants or contain execution caches (`__pycache__`) and should be disregarded. All active source code resides within `src/`.

---

## Key Files for Technical Review

| File / Notebook                                                                                                                                     | What to Look For                                                                                                                       |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| [notebooks/training/](notebooks/training/)                                                                                                             | Full ML pipeline: EDA, LDA, data prep, single-model and multi-model training, survival feature integration, experiment runner          |
| [notebooks/results/](notebooks/results/)                                                                                                               | Results aggregation from SQLite, accuracy/F1/AUC tables, confusion matrices, ROC curves, feature importance, balance strategy rankings |
| [notebooks/eda/](notebooks/eda/)                                                                                                                       | Feature distributions, class imbalance, DTP trends, payment behavior patterns, correlation heatmaps                                    |
| [src/modules/exploratory_data_analysis/linear_discriminant_analysis.py](src/modules/exploratory_data_analysis/linear_discriminant_analysis.py)         | LDA exploratory tool; LD1 explains 79.5% of separation variance                                                                        |
| [src/modules/exploratory_data_analysis/enrollment_statistics.py](src/modules/exploratory_data_analysis/enrollment_statistics.py)                       | Enrollment statistics analysis                                                                                                         |
| [src/utils/data_loaders/](src/utils/data_loaders/)                                                                                                     | Data ingestion layer for revenues, enrollees, and bad debts source files                                                               |
| [src/utils/pseudonymizer.py](src/utils/pseudonymizer.py)                                                                                               | Pseudonymization logic (also exposed as a notebook via `notebooks/Pseudonymizer.ipynb`)                                              |
| [src/modules/machine_learning/models/base_pipeline.py](src/modules/machine_learning/models/base_pipeline.py)                                           | Abstract pipeline design; all classifiers inherit from this                                                                            |
| [src/modules/machine_learning/models/two_stage_classifier.py](src/modules/machine_learning/models/two_stage_classifier.py)                             | Two-stage stacking implementation                                                                                                      |
| [src/modules/machine_learning/models/ordinal_classifier.py](src/modules/machine_learning/models/ordinal_classifier.py)                                 | Ordinal wrapper around base classifiers                                                                                                |
| [src/modules/machine_learning/utils/features/generate_survival_features.py](src/modules/machine_learning/utils/features/generate_survival_features.py) | Cox PH model fitting and survival feature generation                                                                                   |
| [src/modules/machine_learning/utils/balancing/hybrid_resampler.py](src/modules/machine_learning/utils/balancing/hybrid_resampler.py)                   | Custom hybrid undersampling + SMOTE strategy                                                                                           |
| [src/modules/machine_learning/utils/training/run_models_parallel.py](src/modules/machine_learning/utils/training/run_models_parallel.py)               | Windows-compatible parallel experiment execution                                                                                       |
| [src/modules/machine_learning/utils/io/results_repository.py](src/modules/machine_learning/utils/io/results_repository.py)                             | SQLite repository layer for results storage                                                                                            |
| [src/modules/feature_engineering/credit_sales_machine_learning.py](src/modules/feature_engineering/credit_sales_machine_learning.py)                   | Full feature engineering pipeline from raw revenues + enrollees                                                                        |
| [run_experiment.py](run_experiment.py)                                                                                                                 | Papermill headless runner; produces timestamped logs                                                                                   |
| [src/modules/machine_learning/parameters.json](src/modules/machine_learning/parameters.json)                                                           | Hyperparameter grids for all 15 model types                                                                                            |

---

## Setup & Reproduction

### Prepare Required Folders

Before running the pipeline, make sure the following folders exist in your local project root:

- `results/`
- `data/logs/`
- `data/training_input/` (or use `database/` as the source)

These paths are used by the notebooks, experiment runner, and exports. Any raw database exports should be placed locally and must remain excluded from version control.

### Requirements

- Python 3.10+
- CUDA-capable GPU recommended (for XGBoost GPU mode)
- Conda (preferred) or pip

### Install

```bash
# Conda (recommended — includes all pinned versions + CUDA)
conda env create -f environments/environment.yml
conda activate <env_name>

# OR pip
pip install -r environments/requirements.txt
```

### Running Experiments

Before executing experiments, review `settings.json` carefully. This file controls core run boundaries and behavior (including date cutoffs). 

Important: update `observation_end` to match the last available entry in your dataset so that feature engineering, labeling, and experiment windows align with your most recent data.

```bash
# Headless full pipeline run (logs to data/logs/)
python run_experiment.py

# Monitor live output (PowerShell)
Get-Content data/logs/experiment_log_*.txt -Wait
```

### Running the Web App

```bash
python run_app.py
```

### Viewing Results

Open the results notebooks in [`notebooks/results/`](notebooks/results/) and point them to the latest results DB:

```
results/2026_04_18_02/results.db
```

---

## Results Summary (as of April 18, 2026)

- **1092 experiments** completed across all model/strategy/phase combinations
- **Two-stage classifiers** (notably XGB → AdaBoost) show the strongest overall performance
- **Cox PH C-index:** 0.7817 — survival features provide meaningful additional signal
- **Top separating features** (from LDA): `opening_balance_flag`, `opening_balance`, `prev_bracket`, `dtp_wavg`
- Result figures are currently generated to [`data/Results_Base-Graphics-for-Paper/`](data/Results_Base-Graphics-for-Paper/) by the analysis notebooks

---

## Notes for Reviewers

- Neural network models (RNN, Transformer, MLP) were evaluated in an earlier phase and archived in `src/modules/machine_learning/models/_archived/` — they are not part of the final experiment results.
- The `data/_training_results_archived/` folder contains results from runs with known bugs (resampling accumulation, test leakage, schema issues) and should be disregarded. Only `results/2026_04_18_02/` contains valid, bug-corrected results.
- All date fields in the dataset have been shifted/pseudonymized. No real student identifiers are present in any committed file.
