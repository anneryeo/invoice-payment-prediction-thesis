import pandas as pd

class BadDebtsExpense:
    def __init__(self, df_credit_sales):
        self.df_credit_sales = df_credit_sales

    def calculate_bad_debts(self, df_credit_sales):
        df_bde = df_credit_sales[['school_year', 'due_date', 'student_id_pseudonimized', 'net_receivables']].rename(columns={'net_receivables': 'amount_due'})
        df_bde = df_bde.groupby(['school_year', 'due_date', 'student_id_pseudonimized']).sum().reset_index()
        latest_sy = df_bde['school_year'].max()

        # Filter not latest S.Y. and positive values
        df_bde = df_bde[(df_bde['school_year'] != latest_sy) & (df_bde['amount_due'] > 0)]

        # Fill in other columns
        df_bde['entry_date'] = df_bde['due_date'] + pd.DateOffset(days=180)
        df_bde['due_date'] = df_bde['entry_date']
        df_bde['student_id_pseudonimized'] = df_bde['student_id_pseudonimized']
        df_bde['category_name'] = "Bad Debts Expense"
        df_bde['amount_due'] = -df_bde['amount_due']
        df_bde['receivables'] = df_bde['amount_due']

        df_bde.reset_index(inplace=True, drop=True)

        df_bde = df_bde[['entry_date', 'due_date', 'school_year', 'student_id_pseudonimized', 'category_name', 'amount_due', 'receivables']]
            
        return df_bde

    def show_data(self):
        df_bde = self.calculate_bad_debts(self.df_credit_sales)

        return df_bde