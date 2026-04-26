# %%
import os

from pathlib import Path



# Run once at the top — sets cwd to Thesis/

os.chdir(Path().resolve().parents[1])  # eda → notebooks → Thesis


# %%
# Standard libraries

import pandas as pd



# Non-standard Libaries

from src.utils.data_loaders.bad_debts import BadDebtsExpense



from src.modules.feature_engineering.credit_sales_machine_learning import CreditSalesProcessor

from src.modules.feature_engineering.days_sales_outstanding import DSO

from src.modules.feature_engineering.consecutive_years import get_consecutive_years



from src.modules.exploratory_data_analysis.enrollment_statistics import enrollment_statistics


# %%
import os

import sys

from pathlib import Path



# Get the notebook's own path — works in VS Code and via Papermill

_nb_path = Path(

    globals().get("__vsc_ipynb_file__")  # VS Code interactive

    or globals().get("__file__")         # Papermill / script

    or __file__                          # fallback

).resolve()



# Walk up until we find 'src/' (thesis root marker)

def _find_root(start: Path, marker="src") -> Path:

    for parent in [start, *start.parents]:

        if (parent / marker).is_dir():

            return parent

    raise FileNotFoundError(f"Could not find root (no '{marker}' folder found above {start})")



_thesis_root = _find_root(_nb_path)



os.chdir(_thesis_root)

if str(_thesis_root) not in sys.path:

    sys.path.insert(0, str(_thesis_root))



print(f"Working directory: {_thesis_root}")


# %%
from src.utils.data_loaders.read_settings_json import read_settings_json



args = read_settings_json()

args


# %%
df_revenues = pd.read_excel(args['TrainingInput']['REVENUES'], engine='calamine')


# %%
df_enrollees = pd.read_excel(args['TrainingInput']['ENROLLEES'], engine='calamine')

df_enrollees


# %%
df_enrollees = get_consecutive_years(df_enrollees)

df_enrollees[['student_id_pseudonimized','school_year','consecutive_years']]


# %%
cs = CreditSalesProcessor(df_revenues, df_enrollees, args,

                          drop_helper_columns=True,

                          drop_plan_type_columns=True,

                          drop_survival_columns=True,

                          calculate_payment_amounts=True)

df_credit_sales = cs.show_data()


# %%
df_credit_sales


# %%
bde = BadDebtsExpense(df_credit_sales)

df_bde = bde.show_data()

df_bde


# %%
df_chart_of_accounts = pd.read_excel(args['TrainingInput']['CHART_OF_ACCOUNTS'], engine='calamine')


# %%
df_chart_of_accounts


# %%
# Combine bad debts expense to the revenues

# To recognize receivables no longer able for collection

df_all_transactions = pd.concat((df_revenues, df_bde))





df_all_transactions = pd.merge(

    df_all_transactions,

    df_enrollees,

    on=['student_id_pseudonimized', 'school_year'],

    how='inner'

)


# %%
df_all_transactions


# %%
stats = enrollment_statistics(df_enrollees, mode="percent")

stats


# %%
stats = enrollment_statistics(df_enrollees, mode="count")

stats


# %%
df_bde_per_sy = (

    df_bde

    .groupby(["school_year"], as_index=False)

    .agg(bad_debts_expense=("amount_due", "sum"))

)

df_bde_per_sy


# %%
# --- Step 1: Aggregate per school_year + student ---

df_bde_agg = (

    df_bde

    .groupby(["school_year", "student_id_pseudonimized"], as_index=False)

    .agg(bad_debts_expense=("amount_due", "sum"))

)



# --- Step 2: Select only needed columns from df_enrollees ---

df_enr = df_enrollees[["school_year", "student_id_pseudonimized", "consecutive_years"]]



# --- Step 3: Merge aggregated BDE with enrollment info ---

df_merged = pd.merge(

    df_bde_agg,

    df_enr,

    on=["school_year", "student_id_pseudonimized"],

    how="left"   # keeps all BDE records, adds consecutive_years if available

)


# %%
# List of students with BDE that were not enrolled or can no longer be reconciled (uses Back Account as the category)

df_merged[df_merged['consecutive_years'].isna()]


# %%
# --- Step 1: Aggregate the BDE per school_year and per student_id ---

df_bde_agg = (

    df_bde

    .groupby(["school_year", "student_id_pseudonimized"], as_index=False)

    .agg(bad_debts_expense=("amount_due", "sum"))

)



# --- Step 2: Select only needed columns from df_enrollees ---

df_enr = (

    df_enrollees[["school_year", "student_id_pseudonimized", "consecutive_years"]]

)



# --- Step 3: Merge aggregated BDE with enrollment info ---

df_merged = pd.merge(

    df_bde_agg,

    df_enr,

    on=["school_year", "student_id_pseudonimized"],

    how="left"   # keeps all BDE records, adds consecutive_years if available

)



# --- Step 4: Calculate mean consecutive years per school_year ---

df_mean_consec = (

    df_merged

    .groupby("school_year", as_index=False)

    .agg(mean_consecutive_years=("consecutive_years", "mean"))

)



df_mean_consec


# %%
df_merged['consecutive_years'].mean()


# %%
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(10, 6))



# Base violin plot (distribution shape only, no inner marks)

sns.violinplot(

    data=df_merged,

    x="school_year",

    y="consecutive_years",

    inner=None,   # remove inner bars/points

    color="lightgray",  # neutral background

    cut=0

)



# Overlay swarm plot (the "beeswarm" points)

sns.swarmplot(

    data=df_merged,

    x="school_year",

    y="consecutive_years",

    size=3,   # smaller markers

    alpha=0.7,

    color="black"

)



plt.title("Beeswarm Plot: Consecutive Years Enrolled per School Year (Enrolees with Bad Debts Expense)")

plt.xlabel("School Year")

plt.ylabel("Consecutive Years")

plt.tight_layout()

plt.show()


# %%
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(8, 6))



# Add a constant category column for clarity

df_merged["all_years"] = "All School Years"



# Base violin plot (distribution shape only, no inner marks)

sns.violinplot(

    data=df_merged,

    x="all_years",

    y="consecutive_years",

    inner=None,        # remove inner bars/points

    color="lightgray", # neutral background

    cut=0

)



# Overlay swarm plot (the "beeswarm" points)

sns.swarmplot(

    data=df_merged,

    x="all_years",

    y="consecutive_years",

    size=5,

    alpha=0.7,

    color="black"

)



plt.title("Beeswarm Plot: Consecutive Years (All School Years Combined)")

plt.xlabel("")  # no need for x-axis label

plt.ylabel("Consecutive Years")

plt.tight_layout()

plt.show()


# %%
dso = DSO(df_all_transactions, df_credit_sales)

df_dso = dso.show_data()

df_dso['year'] = df_dso['date'].dt.year

df_dso['school_year'] = df_dso['year'] -1

df_dso['month'] = df_dso['date'].dt.month


# %%
# Filter rows where month == June (6) since July is the end of the enrollment period

df_dso_yearly = df_dso[df_dso['date'].dt.month == 6]

df_dso_yearly[['year', 'school_year', 'rolling_12m_dso']]


# %%
df_dso.to_excel("DSO.xlsx", index=False)


# %%
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Filter out 2016, 2017, 2018, 2026

df_plot = df_dso[~df_dso['year'].isin([2016, 2017, 2018, 2026, 2027])]



# Plot with seaborn

plt.figure(figsize=(12, 6))

ax = sns.lineplot(

    data=df_plot,

    x='month',

    y='rolling_12m_dso',

    hue='year',

    marker='o',

    palette='tab10',

    legend=False   # remove legend

)



# Grab the line objects in the same order seaborn plotted them

lines = ax.get_lines()

years = sorted(df_plot['year'].unique())  # ensure consistent ordering



# Add labels at the last entry of each year

for line, year in zip(lines, years):

    group = df_plot[df_plot['year'] == year]

    last_point = group.loc[group['date'].idxmax()]  # last available date for that year

    plt.text(

        x=last_point['month'] + 0.1,   # small offset to the right

        y=last_point['rolling_12m_dso'],

        s=str(year),

        va='center',

        color=line.get_color()  # match line color

    )



# Formatting

plt.title("Rolling 12M DSO by Calendar Year (2019 to 2025)")

plt.xlabel("Month")

plt.ylabel("Rolling 12M DSO")

plt.xticks(range(1, 13), 

           ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

plt.show()


# %%
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Keep only 2022–2025

df_plot = df_dso[df_dso['year'].between(2022, 2025)]



# Plot with seaborn

plt.figure(figsize=(12, 6))

ax = sns.lineplot(

    data=df_plot,

    x='month',

    y='rolling_12m_dso',

    hue='year',

    marker='o',

    palette='tab10',

    legend=False   # remove legend

)



# Grab the line objects in the same order seaborn plotted them

lines = ax.get_lines()

years = sorted(df_plot['year'].unique())



# Add labels at the last entry of each year

for line, year in zip(lines, years):

    group = df_plot[df_plot['year'] == year]

    last_point = group.loc[group['date'].idxmax()]  # last available date for that year

    plt.text(

        x=last_point['month'] + 0.1,   # small offset to the right

        y=last_point['rolling_12m_dso'],

        s=str(year),

        va='center',

        color=line.get_color()

    )



# Formatting

plt.title("Rolling 12M DSO by Calendar Year (2022 to 2025)")

plt.xlabel("Month")

plt.ylabel("Rolling 12M DSO")

plt.xticks(range(1, 13), 

           ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

plt.show()


# %%
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Ensure datetime

df_dso['date'] = pd.to_datetime(df_dso['date'])



# Extract year and month

df_dso['year'] = df_dso['date'].dt.year

df_dso['month'] = df_dso['date'].dt.month



# Keep only 2023–2025

df_plot = df_dso[df_dso['year'].between(2020, 2025)]



# Compute % relative to same year's average

df_plot['year_mean'] = df_plot.groupby('year')['rolling_12m_dso'].transform('mean')

df_plot['relative_pct'] = (df_plot['rolling_12m_dso'] / df_plot['year_mean']) * 100



# Pivot to Year x Month grid

heatmap_data = df_plot.pivot_table(

    index='year',

    columns='month',

    values='relative_pct',

    aggfunc='mean'

)



# Plot heatmap

plt.figure(figsize=(12, 6))

sns.heatmap(

    heatmap_data,

    cmap='coolwarm',       # diverging colormap

    annot=True,

    fmt=".1f",

    linewidths=0.5,

    center=100,            # center at 100% (the yearly average)

    cbar_kws={'label': '% of Yearly Average DSO'}

)



# Formatting

plt.title("Rolling 12M DSO Heatmap (% Relative to Yearly Average, 2023–2025)")

plt.xlabel("Month")

plt.ylabel("Year")

plt.xticks(

    ticks=[i+0.5 for i in range(12)], 

    labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],

    rotation=0

)

plt.yticks(rotation=0)

plt.tight_layout()

plt.show()


# %%
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Ensure datetime

df_dso['date'] = pd.to_datetime(df_dso['date'])



# Extract year and month

df_dso['year'] = df_dso['date'].dt.year

df_dso['month'] = df_dso['date'].dt.month



# Filter out 2016, 2017, 2018, 2026, 2027

df_plot = df_dso[~df_dso['year'].isin([2016, 2017, 2018, 2026, 2027])]



# Plot with seaborn

plt.figure(figsize=(12, 6))

ax = sns.lineplot(

    data=df_plot,

    x='month',

    y='rolling_12m_dso_pct_change',

    hue='year',

    marker='o',

    palette='tab10',

    legend=False   # remove 

)



# Grab the line objects in the same order seaborn plotted them

lines = ax.get_lines()

years = sorted(df_plot['year'].unique())  # ensure consistent ordering



# Add labels at the last entry of each year

for line, year in zip(lines, years):

    group = df_plot[df_plot['year'] == year]

    last_point = group.loc[group['date'].idxmax()]  # last available date for that year

    plt.text(

        x=last_point['month'] + 0.1,   # small offset to the right

        y=last_point['rolling_12m_dso_pct_change'],

        s=str(year),

        va='center',

        color=line.get_color()  # match line color

    )



# Formatting

plt.title("Rolling 12M DSO % Change by Calendar Year (2019 to 2025)")

plt.xlabel("Month")

plt.ylabel("Rolling 12M DSO (% Change)")

plt.xticks(range(1, 13), 

           ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

plt.show()


# %%
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Ensure datetime

df_dso['date'] = pd.to_datetime(df_dso['date'])



# Extract year and month

df_dso['year'] = df_dso['date'].dt.year

df_dso['month'] = df_dso['date'].dt.month



# Keep only 2022–2025

df_plot = df_dso[df_dso['year'].between(2023, 2025)]



# Plot with seaborn

plt.figure(figsize=(12, 6))

ax = sns.lineplot(

    data=df_plot,

    x='month',

    y='rolling_12m_dso_pct_change',

    hue='year',

    marker='o',

    palette='tab10',

    legend=False   # remove legend

)



# Grab the line objects in the same order seaborn plotted them

lines = ax.get_lines()

years = sorted(df_plot['year'].unique())



# Add labels at the last entry of each year

for line, year in zip(lines, years):

    group = df_plot[df_plot['year'] == year]

    last_point = group.loc[group['date'].idxmax()]  # last available date for that year

    plt.text(

        x=last_point['month'] + 0.1,   # small offset to the right

        y=last_point['rolling_12m_dso_pct_change'],

        s=str(year),

        va='center',

        color=line.get_color()

    )



# Formatting

plt.title("Rolling 12M DSO MoM % Change by Calendar Year (2023 to 2025)")

plt.xlabel("Month")

plt.ylabel("Rolling 12M DSO (% Change)")

plt.xticks(range(1, 13), 

           ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

plt.show()


# %%
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Ensure datetime

df_dso['date'] = pd.to_datetime(df_dso['date'])



# Extract year and month

df_dso['year'] = df_dso['date'].dt.year

df_dso['month'] = df_dso['date'].dt.month



# Keep only 2023–2025

df_plot = df_dso[df_dso['year'].between(2023, 2025)]



# Pivot to Year x Month grid

heatmap_data = df_plot.pivot_table(

    index='year',

    columns='month',

    values='rolling_12m_dso_pct_change',

    aggfunc='mean'

)



# Plot heatmap with reversed gradient

plt.figure(figsize=(12, 6))

sns.heatmap(

    heatmap_data,

    cmap='RdYlGn_r',   # reversed colormap

    annot=True,

    fmt=".1f",

    linewidths=0.5,

    vmin=-100,

    vmax=100,

    center=0,

    cbar_kws={'label': 'Rolling 12M DSO % Change'}

)



# Formatting

plt.title("Rolling 12M DSO MoM % Change Heatmap (2023–2025)")

plt.xlabel("Month")

plt.ylabel("Year")

plt.xticks(

    ticks=[i+0.5 for i in range(12)], 

    labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],

    rotation=0

)

plt.yticks(rotation=0)

plt.tight_layout()

plt.show()


# %%
df_plan_a = df_all_transactions[df_all_transactions['plan_type'] == 'Plan - A' ]

df_plan_b = df_all_transactions[df_all_transactions['plan_type'] == 'Plan - B' ]

df_plan_c = df_all_transactions[df_all_transactions['plan_type'] == 'Plan - C' ]



if False:

    cs_a = CreditSales(df_plan_a)

    df_cs_a = cs_a.show_data()



    cs_b = CreditSales(df_plan_b)

    df_cs_b = cs_b.show_data()



    cs_c = CreditSales(df_plan_c)

    df_cs_c = cs_c.show_data()





dso_a = DSO(df_plan_a, df_credit_sales)

df_dso_a = dso_a.show_data()

df_dso_a['year'] = df_dso_a['date'].dt.year

df_dso_a['month'] = df_dso_a['date'].dt.month



dso_b = DSO(df_plan_b, df_credit_sales)

df_dso_b = dso_b.show_data()

df_dso_b['year'] = df_dso_b['date'].dt.year

df_dso_b['month'] = df_dso_b['date'].dt.month



dso_c = DSO(df_plan_c, df_credit_sales)

df_dso_c = dso_c.show_data()

df_dso_c['year'] = df_dso_c['date'].dt.year

df_dso_c['month'] = df_dso_c['date'].dt.month


# %%
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Keep only 2022

def filter_2022(df):

    return df[(df['year'] == 2022)]



df_a_2023 = filter_2022(df_dso_a).assign(source="Plan A")

df_b_2023 = filter_2022(df_dso_b).assign(source="Plan B")

df_c_2023 = filter_2022(df_dso_c).assign(source="Plan C")



# Combine into one dataframe

df_plot = pd.concat([df_a_2023, df_b_2023, df_c_2023])



# Plot with seaborn

plt.figure(figsize=(12, 6))

ax = sns.lineplot(

    data=df_plot,

    x='month',

    y='rolling_12m_dso',

    hue='source',      # color by plan type

    marker='o'

)



# Rename legend title

ax.legend(title="Plan Type")



# Formatting

plt.title("Rolling 12M DSO (2022)")

plt.xlabel("Month")

plt.ylabel("Rolling 12M DSO (in days)")

plt.xticks(

    range(1, 13),

    ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

)

plt.ylim(0, 50)

plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()



plt.savefig(r"data/eda_results/dso_per_plan_2022.png", dpi=300, bbox_inches="tight")

plt.show()


# %%
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Keep only 2023

def filter_2023(df):

    return df[(df['year'] == 2023)]



df_a_2023 = filter_2023(df_dso_a).assign(source="Plan A")

df_b_2023 = filter_2023(df_dso_b).assign(source="Plan B")

df_c_2023 = filter_2023(df_dso_c).assign(source="Plan C")



# Combine into one dataframe

df_plot = pd.concat([df_a_2023, df_b_2023, df_c_2023])



# Plot with seaborn

plt.figure(figsize=(12, 6))

ax = sns.lineplot(

    data=df_plot,

    x='month',

    y='rolling_12m_dso',

    hue='source',      # color by plan type

    marker='o'

)



# Rename legend title

ax.legend(title="Plan Type")



# Formatting

plt.title("Rolling 12M DSO (2023)")

plt.xlabel("Month")

plt.ylabel("Rolling 12M DSO (in days)")

plt.xticks(

    range(1, 13),

    ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

)

plt.ylim(0, 50)

plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()



plt.savefig(r"data/eda_results/dso_per_plan_2023.png", dpi=300, bbox_inches="tight")

plt.show()


# %%
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Keep only 2024

def filter_2024(df):

    return df[(df['year'] == 2024)]



df_a_2024 = filter_2024(df_dso_a).assign(source="Plan A")

df_b_2024 = filter_2024(df_dso_b).assign(source="Plan B")

df_c_2024 = filter_2024(df_dso_c).assign(source="Plan C")



# Combine into one dataframe

df_plot = pd.concat([df_a_2024, df_b_2024, df_c_2024])



# Plot with seaborn

plt.figure(figsize=(12, 6))

ax = sns.lineplot(

    data=df_plot,

    x='month',

    y='rolling_12m_dso',

    hue='source',      # color by plan type

    marker='o'

)



# Rename legend title

ax.legend(title="Plan Type")



# Formatting

plt.title("Rolling 12M DSO (2024)")

plt.xlabel("Month")

plt.ylabel("Rolling 12M DSO (in days)")

plt.xticks(

    range(1, 13),

    ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

)

plt.ylim(0, 50)

plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

plt.savefig(r"data/eda_results/dso_per_plan_2024.png", dpi=300, bbox_inches="tight")

plt.show()


# %%
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Keep only 2025 and restrict to Jan–Oct

def filter_2025(df):

    return df[(df['year'] == 2025)]



df_a_2025 = filter_2025(df_dso_a).assign(source="Plan A")

df_b_2025 = filter_2025(df_dso_b).assign(source="Plan B")

df_c_2025 = filter_2025(df_dso_c).assign(source="Plan C")



# Combine into one dataframe

df_plot = pd.concat([df_a_2025, df_b_2025, df_c_2025])



# Plot with seaborn

plt.figure(figsize=(12, 6))

ax = sns.lineplot(

    data=df_plot,

    x='month',

    y='rolling_12m_dso',

    hue='source',      # color by plan type

    marker='o'

)



# Rename legend title

ax.legend(title="Plan Type")



# Formatting

plt.title("Rolling 12M DSO (2025)")

plt.xlabel("Month")

plt.ylabel("Rolling 12M DSO (in days)")

plt.xticks(

    range(1, 13),  # Jan–Sep only

    ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep', 'Oct', 'Nov', 'Dec']

)

plt.ylim(0, 90)

plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

plt.savefig(r"data/eda_results/dso_per_plan_2025.png", dpi=300, bbox_inches="tight")

plt.show()


# %%
import pandas as pd



students = df_revenues['student_id_pseudonimized'].unique()



# Collect results in a list

results = []



for student in students:

    curr = df_revenues[df_revenues['student_id_pseudonimized'] == student]

    curr_dso = DSO(curr, df_credit_sales)

    df_curr_dso = curr_dso.show_data()

    filtered_dso = df_curr_dso[df_curr_dso['running_receivable'] < 0]

    

    if not filtered_dso.empty:

        # Add student ID as a column for clarity

        filtered_dso = filtered_dso.copy()

        filtered_dso['student_id_pseudonimized'] = student

        results.append(filtered_dso)



# Concatenate all filtered results into one DataFrame

if results:

    df_output = pd.concat(results, ignore_index=True)

    # Export to Excel

    #df_output.to_excel("negative_receivables.xlsx", index=False)


# %%
def label_if_on_time(row):

    row['total_late_payments'] = \

        row['30_days'] \

        + row['60_days'] \

        + row['90_days'] \

        + row['120_days'] \

        + row['150_days'] \

        + row['180_days'] \

        + row['180_above']



    if row['net_receivables'] > 0:

        label = "Not Fully Paid Yet"

    elif row['prepayments'] > 0.00 and row['total_late_payments'] == 0.00:

        label = "On Time"

    else:

        label = "Late"



    return label



df_credit_sales['is_on_time'] = df_credit_sales.apply(label_if_on_time, axis=1)


# %%
import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime



# Filter data

df_cs_already_due = df_credit_sales[df_credit_sales["due_date"] <= datetime.today()]



# Define palette

palette = {

    'On Time': '#00b4d8', # blue

    'Late': '#fb8b24', # red

    'Not Fully Paid Yet': '#495057' # grey

}



# Plot with seaborn (seaborn will count automatically)

plt.figure(figsize=(6,4))

ax = sns.countplot(

    data=df_cs_already_due,

    x='is_on_time',

    palette=palette,

    order=['On Time', 'Late', 'Not Fully Paid Yet']  # fixed order

)



# Compute percentages for annotation

counts = df_cs_already_due['is_on_time'].value_counts()

total = counts.sum()



for i, cat in enumerate(['On Time', 'Late', 'Not Fully Paid Yet']):

    if cat in counts:

        count = counts[cat]

        percent = count / total * 100

        ax.text(i, count + counts.max()*0.01,

                f"{count} ({percent:.1f}%)",

                ha='center', va='bottom', fontsize=10, fontweight='bold')



# Titles and labels

plt.title("On-Time vs Late Credit Sales", fontsize=14, fontweight='bold')

plt.xlabel("Is On Time?", fontsize=12)

plt.ylabel("Count", fontsize=12)

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()


# %%
import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd



# Filter for 2019–2025

# Filter not yet due

df_filtered = df_credit_sales[

    (df_credit_sales['school_year'] >= 2019) &

    (df_credit_sales['school_year'] <= 2025) &

    (df_credit_sales['due_date'] <= datetime.today())

]







# Group by school_year and is_on_time

counts = (

    df_filtered

    .groupby(['school_year', 'is_on_time'])

    .size()

    .reset_index(name='count')

)



# Compute percentages within each school_year

counts['percent'] = (

    counts.groupby('school_year')['count']

          .transform(lambda x: x / x.sum() * 100)

)



# Explicit category order

cat_order = ['On Time', 'Late', 'Not Fully Paid Yet']

counts['is_on_time'] = pd.Categorical(counts['is_on_time'], categories=cat_order, ordered=True)



# Define explicit color mapping

palette = {

    'On Time': '#00b4d8', # blue

    'Late': '#fb8b24', # red

    'Not Fully Paid Yet': '#495057' # grey

}



# Plot with percent on y-axis

plt.figure(figsize=(8,5))

ax = sns.barplot(

    data=counts,

    x='school_year',

    y='percent',

    hue='is_on_time',

    hue_order=cat_order,

    palette=palette

)



# Add labels (percentages)

for i, row in counts.iterrows():

    x_pos = list(counts['school_year'].unique()).index(row['school_year'])

    if row['is_on_time'] == 'On Time':

        hue_offset = -0.25

    elif row['is_on_time'] == 'Late':

        hue_offset = 0

    else:

        hue_offset = 0.25

    ax.text(x_pos + hue_offset, row['percent'] + 1,

            f"{row['percent']:.1f}%", 

            ha='center', va='bottom', fontsize=9, fontweight='bold')



# Titles and labels

plt.title("Credit Sales Status by School Year (2019–2025)", fontsize=14, fontweight='bold')

plt.xlabel("School Year", fontsize=12)

plt.ylabel("Percentage (%)", fontsize=12)

plt.legend(title="Status", loc='upper right')

plt.ylim(0, 100)

plt.tight_layout()

plt.show()


# %%
def label_max_days(row):

    row['total_late_payments'] = \

        row['30_days'] \

        + row['60_days'] \

        + row['90_days'] \

        + row['120_days'] \

        + row['150_days'] \

        + row['180_days'] \

        + row['180_above']



    if row['180_above'] > 0:

        label = "180_above"

    elif row['180_days'] > 0.00:

        label = "180_days"

    elif row['150_days'] > 0.00:

        label = "150_days"

    elif row['120_days'] > 0.00:

        label = "120_days"

    elif row['90_days'] > 0.00:

        label = "90_days"

    elif row['60_days'] > 0.00:

        label = "60_days"

    elif row['30_days'] > 0.00:

        label = "30_days"

    else:

        label = "On-Time"



    return label





# Filter data

df_cs_already_due = df_credit_sales[df_credit_sales["due_date"] <= datetime.today()]

df_cs_already_due['is_on_time'] = df_cs_already_due.apply(label_max_days, axis=1)



df_cs_already_due


# %%
import seaborn as sns

import matplotlib.pyplot as plt



# Compute counts and percentages

label_counts = df_cs_already_due['is_on_time'].value_counts()

label_percentages = (label_counts / label_counts.sum()) * 100



# Reset index for seaborn

df_plot = label_percentages.reset_index()

df_plot.columns = ['Payment Status', 'Percentage']



# Original order from your if-statements

original_order = [

    "180_above",

    "180_days",

    "150_days",

    "120_days",

    "90_days",

    "60_days",

    "30_days",

    "On-Time"

]



# Reverse the order

reversed_order = list(reversed(original_order))



# Filter to only categories that exist in the data

reversed_order = [label for label in reversed_order if label in df_plot['Payment Status'].values]



# Plot with seaborn

plt.figure(figsize=(8, 5))

ax = sns.barplot(

    data=df_plot,

    x='Payment Status',

    y='Percentage',

    order=reversed_order,

    palette='Blues_d',

    hue='Payment Status'

)



# Add percentage labels on top of bars

for p in ax.patches:

    ax.text(

        p.get_x() + p.get_width() / 2,

        p.get_height(),

        f"{p.get_height():.1f}%",

        ha='center',

        va='bottom',

        fontsize=10

    )



# Formatting

ax.set_ylabel("% of invoices")

ax.set_xlabel("Payment status")

ax.set_title("When do invoices get fully paid?")

plt.xticks(rotation=45)

plt.tight_layout()



plt.savefig(r"Images/invoices_payment_status.png", dpi=300, bbox_inches="tight")

plt.show()

