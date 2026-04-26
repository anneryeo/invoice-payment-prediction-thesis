# %%
# Standard libraries

import pandas as pd


# %%
import os

import sys

from pathlib import Path



_here = Path("notebooks/training/Machine Learning.ipynb").resolve().parents[3]

os.chdir(_here)

sys.path.insert(0, str(_here))



print(f"Working directory: {Path().resolve()}")


# %%
from src.utils.data_loaders.read_settings_json import read_settings_json



args = read_settings_json()

args


# %%
df_revenues = pd.read_excel(args['TrainingInput']['REVENUES'], engine='calamine')


# %%
df_revenues


# %%
df_enrollees = pd.read_excel(args['TrainingInput']['ENROLLEES'], engine='calamine')


# %%
df_enrollees


# %%
from src.modules.feature_engineering.credit_sales_machine_learning import CreditSalesProcessor



cs = CreditSalesProcessor(df_revenues, df_enrollees, args)

df_credit_sales = cs.show_data()


# %%
df_credit_sales


# %%
# ── 1. Load settings ────────────────────────────────────────────────────────────────────────

from datetime import datetime

from types import SimpleNamespace

from src.utils.data_loaders.read_settings_json import read_settings_json



settings = read_settings_json(file_path="settings.json")

observation_end = datetime.strptime(settings['Training']['observation_end'], "%Y/%m/%d")

target_feature = settings["Training"]["target_feature"]

test_size = float(settings["Training"]["test_size"])



# SimpleNamespace is picklable by multiprocessing workers (class defined in __main__ is not).

args = SimpleNamespace(

    observation_end = observation_end,

    target_feature  = target_feature,

    test_size       = test_size,                                          # Test size in %

    parameters_dir  = settings['Training']['MODEL_PARAMETERS'],

)







# ── 2. Load the invoice dataset ─────────────────────────────────────────────────────────

from src.modules.feature_engineering.credit_sales_machine_learning import CreditSalesProcessor



cs = CreditSalesProcessor(

    df_revenues, df_enrollees, args,

    drop_demographic_columns=True,

    drop_fully_paid_invoices=False,

    drop_helper_columns=True,

    drop_missing_dtp=True,

    add_streak_features=True,

    exclude_school_years=[2016, 2017, 2018],

    winsorise_dtp=True)

df_credit_sales = cs.show_data()







# ── 3. Load the dataset for machine learning and for survival analysis ───────────────────

survival_columns = ['days_elapsed_until_fully_paid', 'censor']

non_survival_columns = ['due_date', 'dtp_bracket']





df_data = df_credit_sales[df_credit_sales['censor'] == 1].copy()

df_data.drop(columns=survival_columns, inplace=True)



df_data_surv = df_credit_sales.drop(columns=non_survival_columns)







# ── 4. Cox Best Parameters (hardcoded from prior tuning run) ──────────────────────────────

from src.modules.machine_learning.utils.features.adjust_survival_time_periods import adjust_payment_period

from src.modules.machine_learning.utils.features.get_slope_time_points import get_slope_timepoints



X_surv = df_data_surv.drop(columns=survival_columns)

T      = adjust_payment_period(df_data_surv["days_elapsed_until_fully_paid"])   

E      = df_data_surv["censor"]



# Best parameters confirmed by prior tuning run (C-index: 0.7817)

# Hardcoded to skip the 90-fit CV sweep and save ~3–5 minutes per run

best_surv_parameters = {"alpha": 0.05, "l1_ratio": 0.5}

best_time_points = get_slope_timepoints(T, E, n_points=9)

args.time_points = best_time_points



class _TunerStub:

    best_params_  = best_surv_parameters

    best_c_index_ = 0.7817



tuner = _TunerStub()







# ── 5. Data Preparation ──────────────────────────────────────────────────────────────

from src.modules.machine_learning.utils.data.data_preparation import DataPreparer



preparer = DataPreparer(

    df_data,

    target_feature=args.target_feature,

    test_size=args.test_size

)

preparer.encode_labels().train_test_split().resample(balance_strategy="smote_tomek")



X_train = preparer.X_train

X_test = preparer.X_test

y_train = preparer.y_train

y_test = preparer.y_test







# ── 6. Generate survival features ──────────────────────────────────────────────────────────────

from src.modules.machine_learning.utils.features.generate_survival_features import generate_survival_features



X_survival_train, X_survival_test = generate_survival_features(

    X_surv, T, E, X_train, X_test,

    best_params=best_surv_parameters,

    time_points=best_time_points

)



# %%
model_parameters = {

    "learning_rate": 0.1,

    "n_estimators": 50,

    "random_state": 42

}


# %%
from src.modules.machine_learning import AdaBoostPipeline



pipeline = AdaBoostPipeline(

    X_train, X_test, y_train, y_test,

    args,

    model_parameters

)



# Capture results from pipeline

result = pipeline.initialize_model().fit(use_feature_selection=True).evaluate().show_results()

selected_features = pipeline.features



print(f'Accuracy: {result['accuracy']}')

print(f'Precission: {result['precision_macro']}')

print(f'Recall: {result['recall_macro']}')

print(f'F1: {result['f1_macro']}')

print(f'AUC: {result['roc_auc_macro']}')


# %%
selected_features.weights


# %%
model_parameters = {

    "stage1": {

        "colsample_bytree": 0.8,

        "learning_rate": 0.01,

        "max_depth": 3,

        "min_child_weight": 3,

        "n_estimators": 300,

        "reg_alpha": 0.0,

        "reg_lambda": 1.0,

        "subsample": 0.8

    },

    "stage2": {

        "learning_rate": 0.1,

        "n_estimators": 50

    }

}


# %%
from src.modules.machine_learning import TwoStagePipeline

from xgboost import XGBClassifier

from sklearn.ensemble import AdaBoostClassifier



pipeline = TwoStagePipeline(

    X_survival_train, X_survival_test, y_train, y_test,

    args,

    stage1_estimator=XGBClassifier(**model_parameters["stage1"]),

    stage2_estimator=AdaBoostClassifier(**model_parameters["stage2"]),

    use_lda = [True, False],

    lda_mode = "append",

)



result = pipeline.initialize_model().fit(use_feature_selection=True).evaluate().show_results()

selected_features = pipeline.features


# %%
result.keys()


# %%
print(f'Accuracy: {result['accuracy']}')

print(f'Precission: {result['precision_macro']}')

print(f'Recall: {result['recall_macro']}')

print(f'F1: {result['f1_macro']}')

print(f'AUC: {result['roc_auc_macro']}')


# %%
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



# Confusion matrix (convert to NumPy array)

cm = np.array(result['confusion_matrix'])



# Class mapping

class_mapping = {'on_time': 0, '30_days': 1, '60_days': 2, '90_days': 3}

labels = list(class_mapping.keys())



# Normalize by row (true class)

cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]



# Create annotations with both raw counts and percentages

annot = np.empty_like(cm).astype(str)

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        annot[i, j] = f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})"



# Plot heatmap using normalized values for color

plt.figure(figsize=(8,6))

sns.heatmap(cm_normalized, annot=annot, fmt="", cmap="Blues",

            xticklabels=labels, yticklabels=labels)



plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix (Counts + Percentages)")

plt.show()


# %%
selected_features


# %%
assert selected_features.weights is not None

df = pd.DataFrame.from_dict(selected_features.weights['stage_1'], orient='index', columns=['value'])

df['value'].sum()


# %%
from src.modules.machine_learning import (

    AdaBoostPipeline,

    DecisionTreePipeline,

    GaussianNaiveBayesPipeline,

    KNearestNeighborPipeline,

    RandomForestPipeline,

    XGBoostPipeline,

    OrdinalPipeline,

    TwoStagePipeline,

)



models = {

    "ada_boost":           AdaBoostPipeline,

    "decision_tree":       DecisionTreePipeline,

    "gaussian_naive_bayes": GaussianNaiveBayesPipeline,

    "knn":                 KNearestNeighborPipeline,

    "random_forest":       RandomForestPipeline,

    "xgboost":             XGBoostPipeline,

    "ordinal":             OrdinalPipeline,    # expands to ordinal_xgboost, ordinal_random_forest, ordinal_ada_boost

    "two_stage":           TwoStagePipeline,   # expands to 6 xgb/rf/ada combinations

}



# XGBoost uses GPU acceleration — must run sequentially to avoid CUDA context

# conflicts in the thread pool. The runner auto-extends this list to include

# all expanded model names containing 'xgb' (ordinal_xgboost, two_stage_xgb_*,

# two_stage_ada_xgb), so only the base alias is needed here.

do_not_parallel_compute = ['xgboost']



# %%
from src.modules.machine_learning.utils.features.generate_thresholds import generate_thresholds



# Full strategy matrix: no balancing baseline + all SMOTE variants + hybrid threshold sweep

# smote_enn removed — prior runs showed it consistently underperforms borderline_smote

balance_strategies = ["none", "smote", "borderline_smote", "smote_tomek", "hybrid"]



# Hybrid threshold grid: three representative thresholds (threshold choice is a second-order effect)

thresholds = [0.5, 0.7, 0.9]



# %%
# To silence the error when running knn:

# UserWarning: Could not find the number of physical cores for the following reason:

# [WinError 2]

import os



os.environ['OMP_NUM_THREADS'] = '16'


# %%
# Survival related features

drop_columns = ['censor', 'days_elapsed_until_fully_paid']



# Only extract invoices with payments

df_data = df_credit_sales[df_credit_sales['censor'] == 1]



df_data = df_data.drop(columns=drop_columns)



# Drop invoices with missing critical features

df_data.dropna(subset=['dtp_1', 'dtp_2', 'dtp_3', 'dtp_4'], inplace=True)

df_data


# %%
import time

from src.modules.machine_learning.utils.training.run_models_parallel import SurvivalExperimentRunner

from src.modules.machine_learning.utils.io.save_results_to_folder import save_training_results



# Record start time

_train_start = time.time()

_train_start_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(_train_start))



# Create an experiment runner instance

runner = SurvivalExperimentRunner(

    df_data=df_data,

    df_data_surv=df_data_surv,

    models=models,

    balance_strategies=balance_strategies,

    args=args,

    best_parameters=best_surv_parameters,

    thresholds=thresholds,

    n_jobs=-1,

    do_not_parallel_compute=do_not_parallel_compute,

    feature_selection_baseline=True,

    feature_selection_enhanced=True,

    checkpoint_path = os.path.join(settings['Training']['LOGS'], "_checkpoint.pkl"),

)



# Run all experiments — returns (results_df, class_mappings)

df_results, class_mappings = runner.run()



# Record end time

_train_end = time.time()

_train_end_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(_train_end))

_elapsed_s = int(_train_end - _train_start)

_elapsed_str = f"{_elapsed_s // 3600}h {(_elapsed_s % 3600) // 60}m {_elapsed_s % 60}s"

print(f"\nTotal training time: {_elapsed_str}")



# Save results to a new dated folder under Results/

survival_results_dict = {

    "best_c_index":      tuner.best_c_index_,

    "best_parameters":   best_surv_parameters,

    "time_points":       best_time_points,

}



metadata, run_folder = save_training_results(

    model_results_df     = df_results,

    survival_results_dict= survival_results_dict,

    class_mappings_dict  = class_mappings,

    base_output_folder   = settings['Training']['RESULTS_ROOT'],

    model_names          = list(models.keys()),

    start_time           = _train_start_iso,

    end_time             = _train_end_iso,

    total_run_time       = _elapsed_str,

    format               = "sqlite",

)



print(f"Results saved → {run_folder}")



# %%
df_results


# %%
df_results.sort_values(by='enhanced_accuracy', ascending=False)


# %%
df_results.sort_values(by='enhanced_precision_macro', ascending=False)


# %%
df_results.sort_values(by='enhanced_f1_macro', ascending=False)


# %%
df_results.sort_values(by='enhanced_roc_auc_macro', ascending=False)


# %%
import sqlite3

import os

import pandas as pd



# ── Load results from the current run's SQLite database ──────────────────────

# run_folder is set by the training cell above (e.g. "Results\2026_04_18_01")

db_path = os.path.join(run_folder, "results.db")

con = sqlite3.connect(db_path)



df = pd.read_sql_query("""

    SELECT

        e.model,

        e.balance_strategy,

        e.parameters,

        MAX(CASE WHEN m.phase = 'baseline' THEN m.accuracy      END) AS baseline_accuracy,

        MAX(CASE WHEN m.phase = 'baseline' THEN m.f1_macro      END) AS baseline_f1_macro,

        MAX(CASE WHEN m.phase = 'baseline' THEN m.roc_auc_macro END) AS baseline_roc_auc_macro,

        MAX(CASE WHEN m.phase = 'enhanced' THEN m.accuracy      END) AS enhanced_accuracy,

        MAX(CASE WHEN m.phase = 'enhanced' THEN m.f1_macro      END) AS enhanced_f1_macro,

        MAX(CASE WHEN m.phase = 'enhanced' THEN m.roc_auc_macro END) AS enhanced_roc_auc_macro

    FROM experiments e

    JOIN metrics m ON m.experiment_id = e.id

    GROUP BY e.id

""", con)

con.close()



# NOTE: undersample_threshold is not stored in this run's DB.

# Hybrid variants (threshold 0.5 / 0.7 / 0.9) are merged under one 'hybrid' label.

# Their mean score is averaged across all three threshold variants.



# Choose which score column to use

score_column = "enhanced_roc_auc_macro"   # <-- change this to the metric you want



# Compute mean score per model + parameters + balance strategy

grouped = (

    df.groupby(["model", "parameters", "balance_strategy"])[score_column]

    .mean()

    .reset_index()

)



# Rank strategies within each model+parameters group

grouped["rank"] = (

    grouped.groupby(["model", "parameters"])[score_column]

           .rank(method="first", ascending=False)

)



# Assign weighted points: top 1 → 5, top 2 → 4, … top 5 → 1

def assign_points(rank):

    if rank == 1: return 5

    elif rank == 2: return 4

    elif rank == 3: return 3

    elif rank == 4: return 2

    elif rank == 5: return 1

    else: return 0



grouped["points"] = grouped["rank"].apply(assign_points)



# Aggregate across all models to see total points per strategy

strategy_scores = (

    grouped.groupby("balance_strategy")["points"]

    .sum()

    .reset_index()

    .sort_values("points", ascending=False)

)



print("Weighted ranking per model+parameters:")

print(grouped.sort_values(["model", "parameters", "rank"]).to_string())



print("\nGlobal tally of weighted points per balance strategy:")

print(strategy_scores.to_string(index=False))



# %%
import sqlite3

import os

import pandas as pd



# ── Load results from the current run's SQLite database ──────────────────────

# run_folder is set by the training cell above (e.g. "Results\2026_04_18_01")

db_path = os.path.join(run_folder, "results.db")

con = sqlite3.connect(db_path)



df = pd.read_sql_query("""

    SELECT

        e.model,

        e.balance_strategy,

        e.parameters,

        MAX(CASE WHEN m.phase = 'baseline' THEN m.accuracy      END) AS baseline_accuracy,

        MAX(CASE WHEN m.phase = 'baseline' THEN m.f1_macro      END) AS baseline_f1_macro,

        MAX(CASE WHEN m.phase = 'baseline' THEN m.roc_auc_macro END) AS baseline_roc_auc_macro,

        MAX(CASE WHEN m.phase = 'enhanced' THEN m.accuracy      END) AS enhanced_accuracy,

        MAX(CASE WHEN m.phase = 'enhanced' THEN m.f1_macro      END) AS enhanced_f1_macro,

        MAX(CASE WHEN m.phase = 'enhanced' THEN m.roc_auc_macro END) AS enhanced_roc_auc_macro

    FROM experiments e

    JOIN metrics m ON m.experiment_id = e.id

    GROUP BY e.id

""", con)

con.close()



# NOTE: undersample_threshold is not stored in this run's DB.

# Hybrid variants (threshold 0.5 / 0.7 / 0.9) are merged under one 'hybrid' label.

# Their mean score is averaged across all three threshold variants.



# Choose which score column to use

score_column = "enhanced_f1_macro"   # <-- change this to the metric you want



# Compute mean score per model + parameters + balance strategy

grouped = (

    df.groupby(["model", "parameters", "balance_strategy"])[score_column]

    .mean()

    .reset_index()

)



# Rank strategies within each model+parameters group

grouped["rank"] = (

    grouped.groupby(["model", "parameters"])[score_column]

           .rank(method="first", ascending=False)

)



# Assign weighted points: top 1 → 5, top 2 → 4, … top 5 → 1

def assign_points(rank):

    if rank == 1: return 5

    elif rank == 2: return 4

    elif rank == 3: return 3

    elif rank == 4: return 2

    elif rank == 5: return 1

    else: return 0



grouped["points"] = grouped["rank"].apply(assign_points)



# Aggregate across all models to see total points per strategy

strategy_scores = (

    grouped.groupby("balance_strategy")["points"]

    .sum()

    .reset_index()

    .sort_values("points", ascending=False)

)



print("Weighted ranking per model+parameters:")

print(grouped.sort_values(["model", "parameters", "rank"]).to_string())



print("\nGlobal tally of weighted points per balance strategy:")

print(strategy_scores.to_string(index=False))



# %%
from src.modules.feature_engineering.credit_sales_machine_learning import CreditSalesProcessor



cs_test = CreditSalesProcessor(df_revenues, df_enrollees, args,

                      drop_fully_paid_invoices=True,

                      drop_back_account_transactions=True,

                      calculate_payment_amounts=True,

                      add_description=True,

                      drop_missing_dtp=False)

df_cs_test = cs_test.show_data()

df_cs_test


# %%

