import glob
import os
import pandas as pd
import asyncio
import nest_asyncio
nest_asyncio.apply()    # overriding jupyter built in loops
import warnings         # silencing openpyxl warning
import aiofiles
from io import BytesIO

from src.utils.pseudonymizer import Pseudonymizer


class Revenues:
    def __init__(self, revenues_folder, directory):
        self.file_location_revenues = self._list_excel_files(revenues_folder)
        self.directory = directory
        self.drop_columns = ['Level', 'PR#', 'Full Name', 'Prev PR#', 'Particulars',
                             'Check No.', 'ClaimStatus', 'Is Correct', 'Audit Notes']

    # Combine all xlsx files
    def _list_excel_files(self, directory):
        excel_files = glob.glob(os.path.join(directory, "*.xlsx"))
        return excel_files

    def _load_excel_files(self):
        warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
        self.df_revenues = asyncio.run(self._combine_excel_files(self.file_location_revenues,
                                                                self.drop_columns))
        return self.df_revenues

    async def _read_excel_async(self, file, drop_columns):
        async with aiofiles.open(file, 'rb') as f:
            content = await f.read()
        df = pd.read_excel(BytesIO(content), sheet_name='Revenues').drop(columns=drop_columns)
        print(f'Finished loading {file}.')
        
        return df

    async def _combine_excel_files(self, files, drop_columns):
        # Kick off all reads in parallel
        tasks   = [self._read_excel_async(f, drop_columns) for f in files]
        df_list = await asyncio.gather(*tasks)
    
        # Filter out None or empty frames
        clean_dfs = []
        for df in df_list:
            if df is None or df.empty:
                continue
            clean_dfs.append(df)
    
        # If nothing to concat, return an empty DataFrame
        if not clean_dfs:
            result = pd.DataFrame()
        else:
            # Single concat call for all non-empty frames
            result = pd.concat(clean_dfs, ignore_index=True)

    
        return result

    def _data_cleaning(self, df_r):
        # Convert school year, ID No. to integer
        df_r['S.Y.'] = pd.to_numeric(df_r['S.Y.'], errors='coerce').astype('Int64')
        df_r['ID No.'] = pd.to_numeric(df_r['ID No.'], errors='coerce').astype('Int64')
        # Convert 'Amount Due' and 'Amount Paid' from string to numeric
        df_r['Amount Due'] = pd.to_numeric(df_r['Amount Due'], errors='coerce')
        df_r['Amount Paid'] = pd.to_numeric(df_r['Amount Paid'], errors='coerce')
        
        # Data Cleaning
        df_r[['Amount Due', 'Amount Paid']] = df_r[['Amount Due', 'Amount Paid']].fillna(0).round(4)
        df_r = df_r.dropna(subset=['Category']).copy()  # Create a copy to avoid SettingWithCopyWarning
        df_r['Date'] = pd.to_datetime(df_r['Date'], errors='coerce')
        df_r['Due On'] = pd.to_datetime(df_r['Due On'], errors='coerce')

        return df_r

    def _update_due_on(self, df_original, df_replacement):
        # Create a dictionary for quick lookup
        lookup = {}
        for _, row_repl in df_replacement.iterrows():
            school_year = row_repl['School Year']
            plan_type = row_repl['Plan Type']
            due_on = row_repl['Due On']
            if school_year not in lookup:
                lookup[school_year] = {}
            lookup[school_year][plan_type] = due_on
    
        # Update the due dates using the lookup dictionary
        for i, row_orig in df_original.iterrows():
            school_year = row_orig['S.Y.']
            category = str(row_orig['Category'])
            discount_applied = str(row_orig['Discount/Refund Applied To'])

            # Skip if school_year not in lookup
            if school_year not in lookup:
                continue

            for plan_type, due_on in lookup[school_year].items():
                orig_due_on = row_orig['Due On']
                orig_amount_due = row_orig['Amount Due']
                orig_amount_paid = row_orig['Amount Paid']

                if plan_type not in category and plan_type not in discount_applied:
                    continue

                if orig_amount_paid != 0 and orig_due_on < due_on:
                    df_original.at[i, 'Due On'] = due_on
                elif orig_amount_due != 0:
                    df_original.at[i, 'Due On'] = due_on
                else:
                    # No assignment, effectively "continue"
                    pass

                break  # Exit the loop once a match is found
    
        return df_original

    def _feature_engineering(self, df_r):
        # Calculate receivables
        df_r['Receivables'] = df_r['Amount Due'] - df_r['Amount Paid']
        
        return df_r
    
    def _pseudonymize(self, df_r) -> pd.DataFrame:
        p = Pseudonymizer()
       
        df_r_pseudo = p.pseudonymize(df_r)

        return df_r_pseudo
    
    def _rename_columns(self, df_r):
        # Define a mapping from old column names to new column names
        rename_map = {
            "Date": "entry_date",
            "Due On": "due_date",
            "S.Y.": "school_year",
            "ID No.": "student_id_pseudonimized",
            "Category": "category_name",
            "Discount/Refund Applied To": "discount_refund_applied_to",
            "Amount Due": "amount_due",
            "Amount Paid": "amount_paid",
            "Account Name": "account_name",
            "Receivables": "receivables"
        }
        
        # Apply the mapping
        return df_r.rename(columns=rename_map)
    
    def _remove_account_details(self, df_r):
        account_name_map = {
            'Cash on Hand': 'Cash',
            'BDO Unibank - 0102-2800-2477': 'Bank',
            'China Bank': 'Bank',
            'BPI': 'Bank',
            'PS Bank': 'Bank',
            'Metro Bank': 'Bank',
            'PC Bank': 'Bank',
            'Union Bank': 'Bank',
            'Gcash - 0920-284-1954': 'G-Cash',
            'cheque': 'Bank',
            'bank': 'Bank',
            'PNB': 'Bank',
            'China': 'Bank',
            'BPI cheque': 'Bank',
            'PNB ': 'Bank',
            'Bank': 'Bank',
            'Bank (Non-School)': 'Bank',
            'Petty Cash Fund': 'Cash',
            '    ': 'Not Applicable',
            '': 'Not Applicable'
        }

        df_r['Account Name'] = df_r['Account Name'].replace(account_name_map)
        df_r['Account Name'] = df_r['Account Name'].fillna('Not Applicable')

        return df_r
    
    def show_data(self):
        df_r = self._load_excel_files()
        df_r = self._data_cleaning(df_r)
        df_dd = pd.read_excel(self.directory['file_location_settings'], sheet_name="Due Dates")
        df_r = self._update_due_on(df_r, df_dd)
        df_r = self._feature_engineering(df_r)
        df_r = self._pseudonymize(df_r)
        df_r = self._remove_account_details(df_r)
        df_r = self._rename_columns(df_r)

        return df_r