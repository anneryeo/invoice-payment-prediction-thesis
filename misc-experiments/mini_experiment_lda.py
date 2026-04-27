import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from types import SimpleNamespace

# Ensure repo root is in path
sys.path.insert(0, os.path.abspath("."))

from src.utils.data_loaders.read_settings_json import read_settings_json
from src.modules.feature_engineering.credit_sales_machine_learning import CreditSalesProcessor
from src.modules.machine_learning.utils.training.run_models_parallel import SurvivalExperimentRunner
from src.modules.machine_learning import RandomForestPipeline

def run_mini_lda_test():
    print("[test] Starting Mini LDA Experiment...")
    
    # 1. Load settings
    settings = read_settings_json("settings.json")
    obs_end = datetime.strptime(settings["Training"]["observation_end"], "%Y/%m/%d")
    
    args = SimpleNamespace(
        observation_end=obs_end,
        target_feature=settings["Training"]["target_feature"],
        test_size=float(settings["Training"]["test_size"]),
        parameters_dir=settings["Training"]["MODEL_PARAMETERS"],
        time_points=[30, 60, 90, 120] # Shortened for mini test
    )

    # 2. Load data
    print("[data] Loading data...")
    df_rev = pd.read_excel("database/revenues_pseudonymized.xlsx")
    df_enr = pd.read_excel("database/enrollees_pseudonymized.xlsx")
    
    cs = CreditSalesProcessor(
        df_rev, df_enr, args,
        drop_missing_dtp=True,
        drop_demographic_columns=True,
        drop_helper_columns=True,
        exclude_school_years=[2016, 2017, 2018]
    )
    df_full = cs.show_data()
    
    # Prepare classifier vs survival sets
    df_classifier = df_full[df_full["censor"] == 1].copy().drop(columns=["days_elapsed_until_fully_paid", "censor"])
    df_surv = df_full.drop(columns=["due_date", "dtp_bracket"])

    # 3. Setup Runner
    # We use a single model and single strategy for the mini test
    models = {"random_forest": RandomForestPipeline}
    strategies = ["smote"]
    best_params = {"alpha": 0.05, "l1_ratio": 0.5}

    results = []

    for use_lda in [False, True]:
        mode_label = "LDA (3 cols)" if use_lda else "Full (50+ cols)"
        print(f"\n[run] Running Mode: {mode_label}...")
        
        runner = SurvivalExperimentRunner(
            df_data=df_classifier,
            df_data_surv=df_surv,
            models=models,
            balance_strategies=strategies,
            args=args,
            best_parameters=best_params,
            use_lda=use_lda,
            lda_mode="replace", # Pure dimensionality reduction test
            n_jobs=1,
            cache_dir="data/temp_cache/mini_lda"
        )
        
        start_time = time.time()
        res_df, _ = runner.run()
        elapsed = time.time() - start_time
        
        # Extract best F1 from results
        f1_enh = res_df["enhanced_f1_macro"].max()
        f1_base = res_df["baseline_f1_macro"].max()
        
        results.append({
            "Mode": mode_label,
            "Baseline F1": f1_base,
            "Enhanced F1": f1_enh,
            "Time (s)": round(elapsed, 2)
        })

    # 4. Report
    print("\n" + "="*50)
    print("MINI LDA EXPERIMENT RESULTS")
    print("="*50)
    report_df = pd.DataFrame(results)
    print(report_df.to_string(index=False))
    print("="*50)
    
    savings = 100 * (1 - (results[1]["Time (s)"] / results[0]["Time (s)"]))
    print(f"Time Savings: {savings:.1f}%")

if __name__ == "__main__":
    run_mini_lda_test()
