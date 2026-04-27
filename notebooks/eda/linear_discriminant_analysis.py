# %%
# Standard libraries

import pandas as pd


# %%
import os

from pathlib import Path



# Run once at the top — sets cwd to Thesis/

os.chdir(Path().resolve().parents[1])  # eda → notebooks → Thesis


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
df_credit_sales.columns


# %%
from src.modules.exploratory_data_analysis.linear_discriminant_analysis import LDAAnalysis



lda_all = LDAAnalysis(

    df_credit_sales,

    bracket_order = ['on_time', '30_days', '60_days', '90_days'],

    output_path   = 'data/eda_results/lda_all_brackets.png',

    title         = 'LDA Analysis — All Brackets (on_time included)',

)

pipe, X_lda, evr, sep_df = lda_all.run()


# %%
from src.modules.exploratory_data_analysis.linear_discriminant_analysis import LDAAnalysis



lda_delinquent = LDAAnalysis(

    df_credit_sales,

    bracket_order = ['30_days', '60_days', '90_days'],

    output_path   = 'data/eda_results/lda_delinquent.png',

    title         = 'LDA Analysis — Delinquent Only (30 / 60 / 90 days)',

)

pipe, X_lda, evr, sep_df = lda_delinquent.run()

