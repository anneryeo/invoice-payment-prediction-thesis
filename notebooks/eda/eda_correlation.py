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
# Get counts

counts = df_credit_sales.dtp_bracket.value_counts()



# Convert to percentages

percentages = counts / counts.sum() * 100



# Combine into one DataFrame

result = pd.DataFrame({

    'count': counts,

    'percentage': percentages.round(2)

})



print(result)


# %%
import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



# Remove those that have no full dtp_history:

df_filtered = df_credit_sales.dropna(subset=['dtp_1', 'dtp_2', 'dtp_3', 'dtp_4'])





# Clean and filter directly on the DataFrame

df_filtered = df_filtered.loc[

    df_filtered['days_elapsed_until_fully_paid']

        .replace("", np.nan)   # replace empty strings with NaN

        .dropna()              # drop NaNs

        .index                 # keep aligned index

]





# Apply numeric filter

df_filtered = df_filtered[

    (df_filtered['days_elapsed_until_fully_paid'] >= -200) &

    (df_filtered['days_elapsed_until_fully_paid'] <= 200)

]



# Convert censor column to categorical with labels

df_filtered["censor_label"] = (

    df_filtered["censor"]

    .map({0: "Still Unpaid", 1: "Fully Paid"})

    .astype("category")   # force categorical type

)





# KDE plot with grouping by categorical censor labels

sns.kdeplot(

    data=df_filtered,

    x="days_elapsed_until_fully_paid",

    hue="censor_label",

    fill=False,

    common_norm=False,

    palette="Set1"

)



plt.title("KDE Plot: Days Elapsed Until Fully Paid (-200 to +200)")

plt.xlabel("Days Elapsed")

plt.ylabel("Density")

plt.show()


# %%
import matplotlib.pyplot as plt

import seaborn as sns



df_filtered = df_credit_sales[df_credit_sales['censor'] == 1]

df_filtered = df_filtered.dropna(subset=['dtp_1', 'dtp_2', 'dtp_3', 'dtp_4'])



# Select relevant columns

cols = ['days_elapsed_until_fully_paid',

        'dtp_1', 'dtp_2', 'dtp_3', 'dtp_4',

        'dtp_avg', 'dtp_wavg', 'dtp_2_trend',

        'dtp_3_trend', 'days_since_last_payment',

        'credit_sale_amount', 'amount_due_cumsum',

        'amount_paid_cumsum', 'opening_balance']



# Compute correlation matrix

corr = df_filtered[cols].corr()



# Plot heatmap

plt.figure(figsize=(10, 8))

sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f")

plt.title("Correlation with Days Elapsed Until Fully Paid")

plt.show()


# %%
import matplotlib.pyplot as plt

import seaborn as sns



df_filtered = df_credit_sales[df_credit_sales['censor'] == 1]

df_filtered = df_filtered.dropna(subset=['dtp_1', 'dtp_2', 'dtp_3', 'dtp_4'])



# Select relevant columns

cols = ['days_elapsed_until_fully_paid', 'opening_balance_flag', 'payment_ratio',

        'dtp_rolling_std', 'dtp_max', 'early_payer_flag',

        'due_month', 'due_quarter']



# Compute correlation matrix

corr = df_filtered[cols].corr()



# Plot heatmap

plt.figure(figsize=(10, 8))

sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f")

plt.title("Correlation with Days Elapsed Until Fully Paid")

plt.show()


# %%
import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



# Filter and clean data

df_filtered = df_credit_sales[df_credit_sales['censor'] == 1]

df_filtered = df_filtered.dropna(subset=['dtp_1', 'dtp_2', 'dtp_3', 'dtp_4'])



# Define groups of columns

groups = {

    "Payment Dynamics": [

        'days_elapsed_until_fully_paid', 'days_since_last_payment',

        'payment_ratio', 'early_payer_flag'

    ],

    "DTP Basics": ['dtp_1', 'dtp_2', 'dtp_3', 'dtp_4'],

    "DTP Aggregates": ['dtp_avg', 'dtp_wavg'],

    "DTP Trends": ['dtp_2_trend', 'dtp_3_trend'],

    "DTP Variability": ['dtp_rolling_std', 'dtp_max'],

    "Financial Amounts": [

        'credit_sale_amount', 'amount_due_cumsum',

        'amount_paid_cumsum', 'opening_balance',

        'opening_balance_flag'

    ],

    "Calendar Features": ['due_month', 'due_quarter']

}



def pairgrid_with_heatmap(df, cols, hue, title):

    # Build grid

    g = sns.PairGrid(df[cols + [hue]], hue=hue, diag_sharey=False)



    # Lower triangle: scatterplots

    g.map_lower(sns.scatterplot, alpha=0.6, s=40, edgecolor='k')



    # Diagonal: KDE plots

    g.map_diag(sns.kdeplot, fill=True)



    # Upper triangle: correlation heatmap

    corr = df[cols].corr()

    for i, col_i in enumerate(cols):

        for j, col_j in enumerate(cols):

            if j > i:  # upper triangle only

                ax = g.axes[i, j]

                c = corr.loc[col_i, col_j]

                # Map correlation [-1,1] to colormap

                ax.set_facecolor(plt.colormaps['coolwarm']((c+1)/2))

                ax.text(0.5, 0.5, f"{c:.2f}", ha='center', va='center', fontsize=12)

                ax.set_xticks([])

                ax.set_yticks([])



    g.add_legend()

    plt.suptitle(title, y=1.02)

    plt.show()



# Generate plots for each group

for group_name, cols in groups.items():

    pairgrid_with_heatmap(df_filtered, cols, 'dtp_bracket', f"{group_name} Features")

