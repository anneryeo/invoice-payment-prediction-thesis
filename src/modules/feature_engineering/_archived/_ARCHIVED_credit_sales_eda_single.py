import pandas as pd

class CreditSales:
    def __init__(self, df_revenues):
        self.df_revenues = df_revenues.drop(columns=['entry_number'])
        self.df_ad = self.calculate_amount_due(self.df_revenues)
        self.df_disc = self.calculate_discounts(self.df_revenues)
        self.df_adj = self.calculate_adjustments(self.df_revenues)
        self.df_cs = self.get_credit_sale_transactions(self.df_ad)
        self.df_dd = self.calculate_due_dates(self.df_revenues)
        self.df_p = self.calculate_payments(self.df_revenues)
        self.df_cs = self.combine_all_columns(self.df_cs, self.df_dd, self.df_p)
        self.df_cs = self.apply_description_function(self.df_cs)

    def calculate_amount_due(self, df_revenues):
        # calculate categories with zero amount due
        df_has_amount_due = df_revenues.groupby(['school_year',
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
    
        df_ad = df_ad.drop(columns=['discount_refund_applied_to', 'amount_paid', 'amount_due'])

        return df_ad

    def calculate_discounts(self, df_revenues):
        df_disc = df_revenues[
            (df_revenues['category_name'].str.contains("Discount")) &
            (df_revenues['discount_refund_applied_to'] != "")
        ]
        df_disc = df_disc\
            .groupby(['school_year', 'student_id_pseudonimized', 'category_name', 'discount_refund_applied_to'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .drop(columns=['amount_paid', 'receivables'])
        
        # Rename for compatibility during merging with credit sales
        df_disc = df_disc.drop(columns=['category_name'])
        df_disc.rename(columns={'discount_refund_applied_to': 'category_name', 'amount_due': 'amount_discounted'}, inplace=True)
        df_disc = df_disc.groupby(['school_year', 'student_id_pseudonimized', 'category_name']).sum(numeric_only=True)
        df_disc = df_disc.reset_index()
    
        return df_disc

    def calculate_adjustments(self, df_revenues):
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
        df_cs = df_ad
        df_cs.rename(columns={'receivables': 'gross_receivables'}, inplace=True)
        df_cs = df_cs.groupby(['school_year', 'student_id_pseudonimized', 'category_name']).sum(numeric_only=True)
        
        df_cs = pd.merge(df_cs, self.df_disc, on=['school_year', 'student_id_pseudonimized', 'category_name'], how='left')
        df_cs = pd.merge(df_cs, self.df_adj, on=['school_year', 'student_id_pseudonimized', 'category_name'], how='left')
        
        # Fill n/a with 0 the calculate adjusted credit sale amount
        df_cs[['amount_discounted', 'adjustments']] = df_cs[['amount_discounted', 'adjustments']].fillna(0)
        df_cs['credit_sale_amount'] = df_cs['gross_receivables']\
            + df_cs['amount_discounted'] \
            + df_cs['adjustments']
        # Filter with zero credit sale
        df_cs = df_cs[df_cs['credit_sale_amount'] != 0]
    
        return df_cs

    def calculate_due_dates(self, df_revenues):
        df_dd = df_revenues[['school_year', 'student_id_pseudonimized', 'category_name', 'amount_due', 'due_date', 'receivables']]
        df_dd = df_dd[(df_dd['amount_due'] != 0) & (df_dd['receivables'] != 0)]
        df_dd = df_dd.groupby(['school_year', 'student_id_pseudonimized', 'category_name']).min()
        df_dd.reset_index(inplace=True)
        df_dd = df_dd.drop(columns=['amount_due', 'receivables'])
        
        return df_dd

    ####################
    # Calculate Payments
    ####################
    def calculate_prepayments(self, row):
        if (row['Days Elapsed'] <= 0):
            return row['amount_paid']
        else:
            return 0
    
    def calculate_1_to_30_days(self, row):
        if (row['Days Elapsed'] >= 1 and row['Days Elapsed'] <= 30):
            return row['amount_paid']
        else:
            return 0
    
    def calculate_31_to_60_days(self, row):
        if (row['Days Elapsed'] >= 31 and row['Days Elapsed'] <= 60):
            return row['amount_paid']
        else:
            return 0
    
    def calculate_61_to_90_days(self, row):
        if (row['Days Elapsed'] >= 61 and row['Days Elapsed'] <= 90):
            return row['amount_paid']
        else:
            return 0
    
    def calculate_91_to_120_days(self, row):
        if (row['Days Elapsed'] >= 91 and row['Days Elapsed'] <= 120):
            return row['amount_paid']
        else:
            return 0
    
    def calculate_121_to_150_days(self, row):
        if (row['Days Elapsed'] >= 121 and row['Days Elapsed'] <= 150):
            return row['amount_paid']
        else:
            return 0
    
    def calculate_151_to_180_days(self, row):
        if (row['Days Elapsed'] >= 151 and row['Days Elapsed'] <= 180):
            return row['amount_paid']
        else:
            return 0
        
    def calculate_more_than_180_days(self, row):
        if (row['Days Elapsed'] > 180):
            return row['amount_paid']
        else:
            return 0

    def calculate_payments(self, df_revenues):
        df_p = df_revenues[['school_year', 'student_id_pseudonimized', 'entry_date', 'category_name', 'amount_paid', 'receivables']]
        df_p = df_p[df_p['receivables'] < 0]
        df_p = pd.merge(self.df_dd, df_p, on=['school_year', 'student_id_pseudonimized', 'category_name'], how='left')
        df_p = df_p.drop(columns=['receivables'])
        df_p.rename(columns={'entry_date': 'payment_date'}, inplace=True)
        df_p = df_p.dropna(subset=['amount_paid'])
        df_p = df_p[df_p['amount_paid'] != 0]
        df_p['amount_paid'] = df_p['amount_paid'].fillna(0)
        
        df_p['Days Elapsed'] = (df_p['payment_date'] - df_p['due_date']).dt.days
        df_p['prepayments'] = df_p.apply(self.calculate_prepayments, axis=1)
        df_p['30_days'] = df_p.apply(self.calculate_1_to_30_days, axis=1)
        df_p['60_days'] = df_p.apply(self.calculate_31_to_60_days, axis=1)
        df_p['90_days'] = df_p.apply(self.calculate_61_to_90_days, axis=1)
        df_p['120_days'] = df_p.apply(self.calculate_91_to_120_days, axis=1)
        df_p['150_days'] = df_p.apply(self.calculate_121_to_150_days, axis=1)
        df_p['180_days'] = df_p.apply(self.calculate_151_to_180_days, axis=1)
        df_p['180_above'] = df_p.apply(self.calculate_more_than_180_days, axis=1)
        df_p['total_payments'] = \
            df_p['prepayments'] \
            + df_p['30_days'] \
            + df_p['60_days'] \
            + df_p['90_days'] \
            + df_p['120_days'] \
            + df_p['150_days'] \
            + df_p['180_days'] \
            + df_p['180_above']
        df_p = df_p.groupby(['school_year', 'student_id_pseudonimized', 'category_name']).sum(numeric_only=True)
        df_p.reset_index(inplace=True)
        df_p = df_p.drop(columns=['amount_paid', 'Days Elapsed'])
        
        return df_p

    #####################
    # Combine all columns
    #####################
    def combine_all_columns(self, df_cs, df_dd, df_p):
        df_cs = pd.merge(df_cs, df_dd, on=['school_year', 'student_id_pseudonimized', 'category_name'], how='left')
        df_cs = pd.merge(df_cs, df_p, on=['school_year', 'student_id_pseudonimized', 'category_name'], how='left')
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