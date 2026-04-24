# Utilizing Machine Learning to Solve the Invoice Payment Prediction Problem (IPPP)

> **Undergraduate Thesis — Bachelor of Science in Data Science**
> R.J.T. Beley · C.J.L. Reyes

---

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

---

## Database Privacy Notice

> **The source database is private and is NOT included in this repository.**

The `database/` folder references Excel exports that were manually downloaded from internal school systems. These files contain sensitive student financial records and must never be committed, pushed, or shared publicly. The `database/` folder is excluded via `.gitignore`.

Anyone replicating this study must supply their own institutional data that can be transformed into the expected schema via the `CreditSalesProcessor` pipeline. See [FEATURE_REFERENCE.md](FEATURE_REFERENCE.md) for the full column reference, feature descriptions, and required data types.

---

## Methodology

### 1. Data & Feature Engineering

Raw data is sourced from two Excel files:

- **Revenues** — itemized receivables, discounts, adjustments, payment dates
- **Enrollees** — student enrollment records per school year

The `CreditSalesProcessor` class (`feature_engineering/credit_sales_machine_learning.py`) merges these sources and produces the core credit sales dataset. Key engineered features include:

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
- Consecutive school-year enrollment streak (`feature_engineering/consecutive_years.py`)

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
| `hybrid_0.3`       | Undersampling + SMOTE (threshold 0.3) |
| `hybrid_0.5`       | Undersampling + SMOTE (threshold 0.5) |
| `hybrid_0.7`       | Undersampling + SMOTE (threshold 0.7) |

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

LDA is used as an exploratory analytical tool (not for final classification):

- 4-class LDA: LD1 explains **79.5%** of separation variance; top separating features are `opening_balance_flag`, `opening_balance (log1p)`, `prev_bracket`, `dtp_wavg`
- 3-class LDA (delinquent only, excluding on-time): LD1 explains **95.9%** of separation variance

### 8. Web Dashboard

A **Dash** (Plotly) web application (`app.py`) provides an interactive frontend:

- Multi-step setup wizard for uploading data and triggering the pipeline
- KPI dashboard with class distribution and payment timelines
- Per-model performance and feature importance views
- Invoice-level drilldown with payment projections
- Audit logs and export options

---

## Repository Structure

```
.
├── app.py                          # Dash web application entry point
├── run_experiment.py               # Headless notebook runner (Papermill)
├── settings.json                   # Experiment configuration
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment (pinned, CUDA 12.9)
│
├── Machine Learning.ipynb          # ** Primary ML notebook **
├── ML_Results_Analysis.ipynb       # ** Results analysis & visualization **
├── Exploratory Data Analysis.ipynb # Full EDA notebook
├── eda_credit_sales.ipynb          # Credit-sales-specific EDA
│
├── feature_engineering/
│   ├── credit_sales_machine_learning.py  # Main feature engineering pipeline
│   ├── consecutive_years.py              # Enrollment streak feature
│   ├── days_sales_outstanding.py         # DSO financial analysis
│   └── credit_sales_eda.py               # EDA helper
│
├── machine_learning/
│   ├── parameters.json             # Hyperparameter grids for all 15 models
│   ├── models/
│   │   ├── base_pipeline.py        # Abstract base class (all models inherit this)
│   │   ├── ada_boost.py
│   │   ├── decision_tree.py
│   │   ├── gaussian_naive_bayes.py
│   │   ├── k_nearest_neighbor.py
│   │   ├── random_forest.py
│   │   ├── xg_boost.py
│   │   ├── ordinal_classifier.py   # Ordinal wrapper (3 variants)
│   │   └── two_stage_classifier.py # Two-stage stacking (6 variants)
│   └── utils/
│       ├── balancing/              # SMOTE variants + hybrid resampler
│       ├── data/                   # Data preparation, encoding, train/test split
│       ├── features/               # Cox PH survival feature generator
│       ├── training/               # Parallel experiment runner
│       └── io/                     # SQLite schema, results repository, file export
│
├── results/
│   └── 2026_04_18_02/             # ✓ Latest valid results (1092 experiments)
│       └── results.db
│
├── docs/
│   ├── IEEE-PaperDraft-1.ipynb
│   └── 202619APRIL-RESULTSGRAPHS/ # Result figures (heatmaps, ROC, F1, importance)
│
├── logs/                           # Papermill execution logs + executed notebooks
├── analysis/                       # Enrollment statistics script
├── app/                            # Dash app components (screens, utils, assets)
├── scripts/                        # Dev/debugging utility scripts
├── tests/                          # Test suite
├── annesnotes/                     # Developer working notes & planning docs
└── database/                       # ⚠ Private — not committed to version control
```

---

## Key Files for Technical Review

| File / Notebook                                                                                                             | What to Look For                                                                                                                       |
| --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| [Machine Learning.ipynb](Machine%20Learning.ipynb)                                                                             | Full ML pipeline: EDA, LDA, data prep, single-model and multi-model training, survival feature integration, experiment runner          |
| [ML_Results_Analysis.ipynb](ML_Results_Analysis.ipynb)                                                                         | Results aggregation from SQLite, accuracy/F1/AUC tables, confusion matrices, ROC curves, feature importance, balance strategy rankings |
| [Exploratory Data Analysis.ipynb](Exploratory%20Data%20Analysis.ipynb)                                                         | Feature distributions, class imbalance, DTP trends, payment behavior patterns, correlation heatmaps                                    |
| [machine_learning/models/base_pipeline.py](machine_learning/models/base_pipeline.py)                                           | Abstract pipeline design; all classifiers inherit from this                                                                            |
| [machine_learning/models/two_stage_classifier.py](machine_learning/models/two_stage_classifier.py)                             | Two-stage stacking implementation                                                                                                      |
| [machine_learning/models/ordinal_classifier.py](machine_learning/models/ordinal_classifier.py)                                 | Ordinal wrapper around base classifiers                                                                                                |
| [machine_learning/utils/features/generate_survival_features.py](machine_learning/utils/features/generate_survival_features.py) | Cox PH model fitting and survival feature generation                                                                                   |
| [machine_learning/utils/balancing/hybrid_resampler.py](machine_learning/utils/balancing/hybrid_resampler.py)                   | Custom hybrid undersampling + SMOTE strategy                                                                                           |
| [machine_learning/utils/training/run_models_parallel.py](machine_learning/utils/training/run_models_parallel.py)               | Windows-compatible parallel experiment execution                                                                                       |
| [machine_learning/utils/io/db_schema.py](machine_learning/utils/io/db_schema.py)                                               | SQLite schema (v3) for results storage                                                                                                 |
| [feature_engineering/credit_sales_machine_learning.py](feature_engineering/credit_sales_machine_learning.py)                   | Full feature engineering pipeline from raw revenues + enrollees                                                                        |
| [run_experiment.py](run_experiment.py)                                                                                         | Papermill headless runner; produces timestamped logs                                                                                   |
| [machine_learning/parameters.json](machine_learning/parameters.json)                                                           | Hyperparameter grids for all 15 model types                                                                                            |

---

## Setup & Reproduction

### Requirements

- Python 3.10+
- CUDA-capable GPU recommended (for XGBoost GPU mode)
- Conda (preferred) or pip

### Install

```bash
# Conda (recommended — includes all pinned versions + CUDA)
conda env create -f environment.yml
conda activate <env_name>

# OR pip
pip install -r requirements.txt
```

### Running Experiments

```bash
# Headless full pipeline run (logs to logs/experiment_log_*.txt)
python run_experiment.py

# Monitor live output (PowerShell)
Get-Content logs/experiment_log_*.txt -Wait
```

### Running the Web App

```bash
python app.py
```

### Viewing Results

Open [ML_Results_Analysis.ipynb](ML_Results_Analysis.ipynb) and point it to the latest results DB:

```
results/2026_04_18_02/results.db
```

---

## Results Summary (as of April 18, 2026)

- **1092 experiments** completed across all model/strategy/phase combinations
- **Two-stage classifiers** (notably XGB → AdaBoost) show the strongest overall performance
- **Cox PH C-index:** 0.7817 — survival features provide meaningful additional signal
- **Top separating features** (from LDA): `opening_balance_flag`, `opening_balance`, `prev_bracket`, `dtp_wavg`
- Result figures stored in [docs/202619APRIL-RESULTSGRAPHS/](docs/202619APRIL-RESULTSGRAPHS/)

---

## Notes for Reviewers

- Neural network models (RNN, Transformer, MLP) were evaluated in an earlier phase and archived in `machine_learning/models/_archived/` — they are not part of the final experiment results.
- The `results/2026_04_16_01/` and `results/2026_04_18_01/` folders contain results from runs with known bugs (resampling accumulation, test leakage, schema issues) and should be disregarded. Only `results/2026_04_18_02/` contains valid, bug-corrected results.
- All date fields in the dataset have been shifted/pseudonymized. No real student identifiers are present in any committed file.
