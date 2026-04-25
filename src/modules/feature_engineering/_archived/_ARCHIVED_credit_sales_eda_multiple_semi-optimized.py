import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

class CreditSales:
    def __init__(self, df_revenues):
        self.df_revenues = df_revenues.drop(columns=['entry_number'])
        self.df_ad = self.calculate_amount_due(self.df_revenues)
        self.df_disc = self.calculate_discounts(self.df_revenues)
        self.df_adj = self.calculate_adjustments(self.df_revenues)
        self.df_cs = self.get_credit_sale_transactions(self.df_ad)
        self.df_cs = self.get_payments(self.df_revenues)
        self.df_cs = self.combine_all_columns(self.df_cs)
        self.df_cs = self.apply_description_function(self.df_cs)

    def calculate_amount_due(self, df_revenues) -> pd.DataFrame:
        # calculate categories with zero amount due
        df_has_amount_due = df_revenues.groupby(['school_year',
                                                 #'due_date',
                                                 'student_id_pseudonimized',
                                                 'category_name']).sum(numeric_only=True)
        df_has_amount_due = df_has_amount_due[df_has_amount_due['amount_due'] == 0]
        df_has_amount_due = df_has_amount_due.reset_index().drop(columns=["amount_due", "amount_paid"])
        
        # get the rows to be removed
        common_rows = pd.merge(df_revenues, df_has_amount_due,
                               on=['school_year', 'student_id_pseudonimized', 'category_name'],
                               how='inner')
        
        # Filter out the common rows from DataFrame A
        df_ad = df_revenues[~df_revenues.set_index(['school_year', 'student_id_pseudonimized', 'category_name']).\
            index.isin(common_rows.set_index(['school_year', 'student_id_pseudonimized', 'category_name']).index)]
        df_ad.reset_index()
        
        
        # Filter non credit sales based on criterias
        df_ad = df_ad.query(
            '`amount_due` != 0 and `receivables` > 0 and `amount_paid` >= 0 and '
            '`category_name` != "Refund" and `category_name` != "Overpayment" and '
            'not `category_name`.str.contains("Discount")'
        )
    
        df_ad = df_ad[['entry_date', 'school_year', 'student_id_pseudonimized', 'category_name', 'due_date', 'receivables']]

        return df_ad

    def calculate_discounts(self, df_revenues) -> pd.DataFrame:
        df_disc = df_revenues[
            (df_revenues['category_name'].str.contains("Discount")) &
            (df_revenues['discount_refund_applied_to'] != "")
        ]


        df_disc = df_disc\
            .groupby(['entry_date', 'school_year', 'student_id_pseudonimized', 'category_name', 'discount_refund_applied_to'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .drop(columns=['amount_paid', 'receivables'])
        
        # Rename for compatibility during merging with credit sales
        df_disc = df_disc.drop(columns=['category_name'])
        df_disc.rename(columns={'discount_refund_applied_to': 'category_name',
                                'amount_due': 'amount_discounted'}, inplace=True)
        df_disc = df_disc.groupby(['entry_date', 'school_year', 'student_id_pseudonimized', 'category_name']).sum(numeric_only=True)
        df_disc = df_disc.reset_index()
    
        return df_disc

    def calculate_adjustments(self, df_revenues) -> pd.DataFrame:
        # 1. Filter and copy to avoid SettingWithCopyWarning
        df_adj = df_revenues.query(
            '`amount_due` < 0 and '
            '`category_name` != "Refund" and '
            'not `category_name`.str.contains("Discount")'
        ).copy()
    
        # 2. Use .loc for safe assignment
        mask = df_adj['discount_refund_applied_to'].notna() & (df_adj['discount_refund_applied_to'] != '')
        df_adj.loc[mask, 'category_name'] = df_adj.loc[mask, 'discount_refund_applied_to']
    
        # 3. Rename and aggregate without inplace operations
        df_adj = (
            df_adj[['school_year', 'student_id_pseudonimized', 'category_name', 'amount_due']]
            .rename(columns={'amount_due': 'adjustments'})
            .groupby(['school_year', 'student_id_pseudonimized', 'category_name'], as_index=False)
            .sum()
        )

        return df_adj

    def get_credit_sale_transactions(self, df_ad):
        df_cs = df_ad.copy()
        df_cs.rename(columns={'receivables': 'gross_receivables'}, inplace=True)

        # Sort receivables chronologically
        df_cs.sort_values(
            by=['school_year', 'student_id_pseudonimized', 'category_name', 'due_date'],
            inplace=True
        )

        # --- Prepare discounts ---
        df_disc = self.df_disc.copy()
        # Ensure 'amount_discounted' column exists
        if 'amount_discounted' not in df_disc.columns:
            df_disc['amount_discounted'] = 0.0

        # --- Prepare adjustments ---
        if hasattr(self, "df_adj"):
            df_adj = (
                self.df_adj.groupby(
                    ['school_year', 'student_id_pseudonimized', 'category_name'], as_index=False
                )['adjustments'].sum()
            )
        else:
            df_adj = pd.DataFrame(columns=['school_year','student_id_pseudonimized','category_name','adjustments'])

        # Convert adjustments to dict for O(1) lookup
        adj_dict = dict(
            zip(
                zip(df_adj['school_year'], df_adj['student_id_pseudonimized'], df_adj['category_name']),
                df_adj['adjustments']
            )
        )

        # Pre-group discounts for faster access
        disc_groups = {
            key: subdf.sort_values('entry_date').to_numpy()
            for key, subdf in df_disc.groupby(['school_year','student_id_pseudonimized','category_name'])
        }

        # Precompute discount column index once
        disc_amount_idx = df_disc.columns.get_loc('amount_discounted')

        # Parallel processing of groups
        grouped = df_cs.groupby(['school_year','student_id_pseudonimized','category_name','due_date'])
        args = ((g, disc_groups, adj_dict, disc_amount_idx) for _, g in grouped)

        with Pool(processes=cpu_count()) as pool:
            results = pool.map(allocate_discount_and_adjustments, args)

        df_cs = pd.concat(results).reset_index(drop=True)
        df_cs = df_cs[df_cs['credit_sale_amount'] != 0]

        return df_cs

    def get_payments(self, df_revenues):
        # --- Prepare payment records ---
        df_p = (
            df_revenues[['school_year', 'student_id_pseudonimized', 'entry_date',
                         'category_name', 'amount_paid', 'receivables']]
            .query("receivables < 0 and amount_paid.notnull() and amount_paid != 0")
            .rename(columns={'entry_date': 'payment_date'})
            .assign(amount_paid=lambda d: d['amount_paid'].fillna(0))
        )

        # MultiIndex including due_date for uniqueness
        df_p.set_index(['school_year', 'student_id_pseudonimized', 'category_name'], inplace=True)
        self.df_cs.set_index(['school_year', 'student_id_pseudonimized', 'category_name'], inplace=True)

        # --- Ensure bucket columns exist ---
        bucket_cols = ['prepayments','30_days','60_days','90_days',
                       '120_days','150_days','180_days','180_above','total_payments']
        for col in bucket_cols:
            if col not in self.df_cs.columns:
                self.df_cs[col] = 0.0

        self.df_cs = self.df_cs.reset_index()

        # --- Build tasks for multiprocessing ---
        tasks = []
        for keys, payments in df_p.groupby(level=[0,1,2]):
            receivables = self.df_cs[
                (self.df_cs['school_year'] == keys[0]) &
                (self.df_cs['student_id_pseudonimized'] == keys[1]) &
                (self.df_cs['category_name'] == keys[2])
            ].sort_values(by='due_date').copy()
            tasks.append((receivables, payments, bucket_cols))

        # --- Run in parallel ---
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(allocate_payments, tasks)

        # --- Update self.df_cs with results ---
        for updated in results:
            self.df_cs.update(updated[bucket_cols])

        return self.df_cs.reset_index()
    
    #####################
    # Combine all columns
    #####################
    def combine_all_columns(self, df_cs):
        # Fill n/a with 0
        numeric_columns = ['prepayments', 'total_payments', '30_days', '60_days',
                         '90_days', '120_days', '150_days', '180_days', '180_above']
        df_cs[numeric_columns] = df_cs[numeric_columns].fillna(0)
        
        df_cs['adjusted_credit_amount'] = df_cs['credit_sale_amount'] - df_cs['prepayments']
        df_cs = df_cs[df_cs['credit_sale_amount'] != 0]
        df_cs['net_receivables'] = df_cs['credit_sale_amount'] - df_cs['total_payments']

        return df_cs

    def get_description(self, row):
        category = row['category_name']
        if '-UE' in category:
            return "Tuition fee (" + category[:3] + ") - Upon enrollment"
        elif 'B-1st' in category:
            return "Tuition fee (" + category[:3] + ") - 1 of 2 payments"
        elif 'B-2nd' in category:
            return "Tuition fee (" + category[:3] + ") - 2 of 2 payments"
        elif 'C-1st' in category:
            return "Tuition fee (" + category[:3] + ") - 1 of 4 payments"
        elif 'C-2nd' in category:
            return "Tuition fee (" + category[:3] + ") - 2 of 4 payments"
        elif 'C-3rd' in category:
            return "Tuition fee (" + category[:3] + ") - 3 of 4 payments"
        elif 'C-4th' in category:
            return "Tuition fee (" + category[:3] + ") - 4 of 4 payments"
        elif 'D-1st' in category:
            return "Tuition fee (" + category[:3] + ") - 1 of 10 payments"
        elif 'D-2nd' in category:
            return "Tuition fee (" + category[:3] + ") - 2 of 10 payments"
        elif 'D-3rd' in category:
            return "Tuition fee (" + category[:3] + ") - 3 of 10 payments"
        elif 'D-4th' in category:
            return "Tuition fee (" + category[:3] + ") - 4 of 10 payments"
        elif 'D-5th' in category:
            return "Tuition fee (" + category[:3] + ") - 5 of 10 payments"
        elif 'D-6th' in category:
            return "Tuition fee (" + category[:3] + ") - 6 of 10 payments"
        elif 'D-7th' in category:
            return "Tuition fee (" + category[:3] + ") - 7 of 10 payments"
        elif 'D-8th' in category:
            return "Tuition fee (" + category[:3] + ") - 8 of 10 payments"
        elif 'D-9th' in category:
            return "Tuition fee (" + category[:3] + ") - 9 of 10 payments"
        elif 'D-10th' in category:
            return "Tuition fee (" + category[:3] + ") - 9 of 10 payments"
        elif 'E-1st' in category:
            return "Tuition fee (" + category[:3] + ") - 1 of 9 payments"
        elif 'E-2nd' in category:
            return "Tuition fee (" + category[:3] + ") - 2 of 9 payments"
        elif 'E-3rd' in category:
            return "Tuition fee (" + category[:3] + ") - 3 of 9 payments"
        elif 'E-4th' in category:
            return "Tuition fee (" + category[:3] + ") - 4 of 9 payments"
        elif 'E-5th' in category:
            return "Tuition fee (" + category[:3] + ") - 5 of 9 payments"
        elif 'E-6th' in category:
            return "Tuition fee (" + category[:3] + ") - 6 of 9 payments"
        elif 'E-7th' in category:
            return "Tuition fee (" + category[:3] + ") - 7 of 9 payments"
        elif 'E-8th' in category:
            return "Tuition fee (" + category[:3] + ") - 8 of 9 payments"
        elif 'E-9th' in category:
            return "Tuition fee (" + category[:3] + ") - 9 of 9 payments"
        elif 'E-Learning' in category:
            return "E-learning platform (" + category[:3] + ")"
        elif '-OF-1st' in category:
            return "Miscellaneous fees - 1 of 3 payments"
        elif '-OF-2nd' in category:
            return "Miscellaneous fees - 2 of 3 payments"
        elif '-OF-3rd' in category:
            return "Miscellaneous fees - 3 of 3 payments"
        elif '-OF' in category:
            return "Miscellaneous fees"
        elif 'Books' in category:
            return "Books (" + category[:3] + ")"
        elif 'Events' in category:
            return category.replace('Events - ', '')
        elif category == 'Disturbance Charges':
            return 'Penalties for late enrollee'
        elif category == 'Locker - Small':
            return 'Locker rental (small)'
        elif category == 'Locker - Big':
            return 'Locker rental (big)'
        elif 'Tutorial' in category:
            return 'Tutoring services'
        elif category == 'Uniform - Students - Daily':
            return 'Daily Uniform'
        elif category == 'Uniform - Students - P.E.':
            return 'P.E. Uniform'
        elif category == 'Uniform - Students - Scouting':
            return 'Scouting Uniform'
        elif 'Moving Up' in category:
            return 'Moving up fee'
        elif category == 'Graduation - Fee':
            return 'Graduation fee'
        elif category == 'Graduation - Others':
            return 'Graduation fee (Miscellaneous)'
        else:
            return category
    
    def apply_description_function(self, df_cs):
        df_cs['description'] = df_cs.apply(self.get_description, axis=1)
        return df_cs

    def show_data(self):
        return self.df_cs

# --- helper function must be top-level for multiprocessing ---
def allocate_discount_and_adjustments(args):
    group, disc_groups, adj_dict, disc_amount_idx = args
    key = (
        group['school_year'].iloc[0],
        group['student_id_pseudonimized'].iloc[0],
        group['category_name'].iloc[0]
    )

    group_discounts = disc_groups.get(key, np.empty((0, 0)))
    if group_discounts.size and disc_amount_idx is not None and disc_amount_idx < group_discounts.shape[1]:
        rem_disc = group_discounts[:, disc_amount_idx].sum()
    else:
        rem_disc = 0.0

    rem_adj = adj_dict.get(key, 0.0)

    gross = group['gross_receivables'].to_numpy()
    disc_applied = np.zeros(len(group))
    adj_applied = np.zeros(len(group))

    for i in range(len(group)):
        rec = gross[i]
        apply_disc = np.sign(rem_disc) * min(abs(rem_disc), rec)
        rem_disc -= apply_disc

        remaining_after_disc = rec - apply_disc
        apply_adj = np.sign(rem_adj) * min(abs(rem_adj), remaining_after_disc)
        rem_adj -= apply_adj

        disc_applied[i] = apply_disc
        adj_applied[i] = apply_adj

    return group.assign(
        amount_discounted = disc_applied,
        adjustments = adj_applied,
        credit_sale_amount = gross + disc_applied + adj_applied
    )

def allocate_payments(args):
    receivables, payments, bucket_cols = args
    
    # Ensure datetime types
    payments['payment_date'] = pd.to_datetime(payments['payment_date'], errors='coerce')
    receivables['due_date'] = pd.to_datetime(receivables['due_date'], errors='coerce')

    rem_pay = payments['amount_paid'].sum()
    payment_date = payments['payment_date'].max()

    # Reset buckets
    receivables.loc[:, bucket_cols[:-1]] = 0.0

    # Sequential allocation (due_date order)
    for i, row in receivables.iterrows():
        rec_amt = row['credit_sale_amount']
        apply_amt = min(rem_pay, rec_amt)
        rem_pay -= apply_amt

        # Safe timedelta calculation
        if pd.notnull(payment_date) and pd.notnull(row['due_date']):
            days = (payment_date - row['due_date']).days
        else:
            days = np.inf  # treat missing dates as very late

        # Bucket allocation
        if days <= 0:
            receivables.at[i, 'prepayments'] += apply_amt
        elif days <= 30:
            receivables.at[i, '30_days'] += apply_amt
        elif days <= 60:
            receivables.at[i, '60_days'] += apply_amt
        elif days <= 90:
            receivables.at[i, '90_days'] += apply_amt
        elif days <= 120:
            receivables.at[i, '120_days'] += apply_amt
        elif days <= 150:
            receivables.at[i, '150_days'] += apply_amt
        elif days <= 180:
            receivables.at[i, '180_days'] += apply_amt
        else:
            receivables.at[i, '180_above'] += apply_amt

    receivables['total_payments'] = receivables[bucket_cols[:-1]].sum(axis=1)
    return receivables