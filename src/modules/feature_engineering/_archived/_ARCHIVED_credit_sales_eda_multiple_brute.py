import pandas as pd
import numpy as np

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
        df_cs = df_cs.sort_values(
            by=['school_year', 'student_id_pseudonimized', 'category_name', 'due_date']
        )

        # --- Prepare discounts (aggregated per entry_date) ---
        df_disc = self.df_disc

        # --- Prepare adjustments (group totals) ---
        if hasattr(self, "df_adj"):
            df_adj = (
                self.df_adj.groupby(['school_year', 'student_id_pseudonimized', 'category_name'], as_index=False)
                ['adjustments'].sum()
            )
        else:
            df_adj = pd.DataFrame(columns=['school_year','student_id_pseudonimized','category_name','adjustments'])

        # --- Allocation with overflow + date sensitivity ---
        def allocate(group):
            disc_applied, adj_applied = [], []

            # pool of discounts: all matching discounts for this group
            group_discounts = df_disc[
                (df_disc['school_year'] == group['school_year'].iloc[0]) &
                (df_disc['student_id_pseudonimized'] == group['student_id_pseudonimized'].iloc[0]) &
                (df_disc['category_name'] == group['category_name'].iloc[0])
            ]

            # adjustments pool
            adj_row = df_adj[
                (df_adj['school_year'] == group['school_year'].iloc[0]) &
                (df_adj['student_id_pseudonimized'] == group['student_id_pseudonimized'].iloc[0]) &
                (df_adj['category_name'] == group['category_name'].iloc[0])
            ]
            rem_adj = float(adj_row['adjustments'].iloc[0]) if not adj_row.empty else 0.0

            # total discount pool
            rem_disc = group_discounts['amount_discounted'].sum()

            for idx, row in group.iterrows():
                rec = row['gross_receivables']

                # find applicable discounts (entry_date <= sale entry_date)
                valid_disc = group_discounts[group_discounts['entry_date'] <= row['entry_date']]
                # pool is already summed, so just apply from rem_disc
                apply_disc = np.sign(rem_disc) * min(abs(rem_disc), rec)
                rem_disc -= apply_disc

                # Remaining after discount
                remaining_after_disc = rec - apply_disc

                # Apply adjustment up to remaining
                apply_adj = np.sign(rem_adj) * min(abs(rem_adj), remaining_after_disc)
                rem_adj -= apply_adj

                disc_applied.append(apply_disc)
                adj_applied.append(apply_adj)

            group['amount_discounted'] = disc_applied
            group['adjustments'] = adj_applied
            group['credit_sale_amount'] = group['gross_receivables'] + group['amount_discounted'] + group['adjustments']

            return group

        df_cs = df_cs.groupby(
            ['school_year', 'student_id_pseudonimized', 'category_name'],
            group_keys=False
        ).apply(allocate)

        df_cs = df_cs[df_cs['credit_sale_amount'] != 0]

        return df_cs

    def get_payments(self, df_revenues):
        # Prepare payment records
        df_p = df_revenues[['school_year', 'student_id_pseudonimized', 'entry_date',
                            'category_name', 'amount_paid', 'receivables']].copy()
        df_p = df_p[df_p['receivables'] < 0]  # only payments
        df_p.rename(columns={'entry_date': 'payment_date'}, inplace=True)
        df_p = df_p.dropna(subset=['amount_paid'])
        df_p = df_p[df_p['amount_paid'] != 0]
        df_p['amount_paid'] = df_p['amount_paid'].fillna(0)

        # --- Allocation function per group (school_year, student_id, category) ---
        def allocate(sy, sid, cat, payments):
            receivables = self.df_cs[
                (self.df_cs['school_year'] == sy) &
                (self.df_cs['student_id_pseudonimized'] == sid) &
                (self.df_cs['category_name'] == cat)
            ].sort_values(by='due_date').copy()

            rem_pay = payments['amount_paid'].sum()

            # Ensure bucket columns exist in self.df_cs
            bucket_cols = ['prepayments','30_days','60_days','90_days',
                        '120_days','150_days','180_days','180_above']
            for col in bucket_cols + ['total_payments']:
                if col not in self.df_cs.columns:
                    self.df_cs[col] = 0.0

            # Initialize in subset
            for col in bucket_cols:
                receivables[col] = 0.0

            # Allocate payments
            for idx, row in receivables.iterrows():
                rec = row['credit_sale_amount']
                apply_amt = min(rem_pay, rec)
                rem_pay -= apply_amt

                payment_date = payments['payment_date'].max()
                days = (payment_date - row['due_date']).days

                if days <= 0:
                    receivables.at[idx, 'prepayments'] += apply_amt
                elif 1 <= days <= 30:
                    receivables.at[idx, '30_days'] += apply_amt
                elif 31 <= days <= 60:
                    receivables.at[idx, '60_days'] += apply_amt
                elif 61 <= days <= 90:
                    receivables.at[idx, '90_days'] += apply_amt
                elif 91 <= days <= 120:
                    receivables.at[idx, '120_days'] += apply_amt
                elif 121 <= days <= 150:
                    receivables.at[idx, '150_days'] += apply_amt
                elif 151 <= days <= 180:
                    receivables.at[idx, '180_days'] += apply_amt
                else:
                    receivables.at[idx, '180_above'] += apply_amt

            receivables['total_payments'] = receivables[bucket_cols].sum(axis=1)

            # Update df_cs with new values
            self.df_cs.update(receivables)

        # Iterate over groups
        for (sy, sid, cat) in self.df_cs[['school_year', 'student_id_pseudonimized', 'category_name']].drop_duplicates().itertuples(index=False):
            payments = df_p[
                (df_p['school_year'] == sy) &
                (df_p['student_id_pseudonimized'] == sid) &
                (df_p['category_name'] == cat)
            ]
            if not payments.empty:
                allocate(sy, sid, cat, payments)

        return self.df_cs
    
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