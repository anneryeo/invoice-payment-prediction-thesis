# %%
import os
from pathlib import Path
import sys

# Automatically find repo root by looking for .git
ROOT = Path.cwd()
while not (ROOT / ".git").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Change the working directory to the repo root
os.chdir(ROOT)

# %%
import pandas as pd

# %%
from src.utils.data_loaders.read_settings_json import read_settings_json

args = read_settings_json()
args

# %%
df_revenues = pd.read_excel(args['TrainingInput']['REVENUES'], engine='calamine')

# %%
df_enrollees = pd.read_excel(args['TrainingInput']['ENROLLEES'], engine='calamine')

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
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

from src.modules.machine_learning.utils.inference.inference_pipeline import (
    find_deployed_model,
    load_inference_pipeline,
    run_batch_inference,
)

MODEL_DIR = args["Training"]["DEPLOYED_MODELS"]

# %%
# Locate and inspect the deployed artifact.
# Raises ValueError with upgrade instructions if the artifact is pre-InferencePipeline.
artifact_path = find_deployed_model(MODEL_DIR)
print("Artifact:", artifact_path)

try:
    pipeline = load_inference_pipeline(MODEL_DIR)
    print(pipeline)
except ValueError as e:
    print(f"[WARN] Could not load InferencePipeline:\n  {e}\n")
    print("Action required: Re-run Step 5 (Model Finalization) in the app to regenerate the artifact.")
    pipeline = None

# %%
if pipeline is not None:
    # Select only numeric ML features from df_cs_test (drop non-numeric / label cols)
    EXCLUDE = {"dtp_bracket", "date_fully_paid", "due_date",
               "school_year", "student_id_pseudonimized", "category_name", "description"}
    X_infer = df_cs_test.drop(columns=[c for c in EXCLUDE if c not in df_cs_test.columns])

    df_preds = run_batch_inference(
        input_source=X_infer,
        model_dir=MODEL_DIR,
        batch_size=1024,
        return_proba=True,
    )
    display(df_preds.head(10))
else:
    df_preds = None
    print("Skipping inference — no valid InferencePipeline loaded.")

# %%
if df_preds is not None:
    print("Predicted label distribution:")
    display(df_preds["predicted_label"].value_counts().rename("count").to_frame())
    n  = len(df_preds)
    mk = df_preds["model_key"].iloc[0]
    ts = df_preds["run_timestamp"].iloc[0]
    print()
    print("Total rows scored:", n)
    print("Model key        :", mk)
    print("Run timestamp    :", ts)


