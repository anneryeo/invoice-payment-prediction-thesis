import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import OneHotEncoder

class CreditSales:
    """
    Organizes and processes student credit sales by due-date type, applying discounts,
    adjustments, and payment allocations into time-based buckets.

    Parameters
    ----------
    df_revenues : pd.DataFrame
        Input DataFrame containing revenue records. Must include the following columns:
        - 'entry_date'
        - 'due_date'
        - 'school_year'
        - 'student_id_pseudonimized'
        - 'category_name'

    Notes
    -----
    The resulting DataFrame includes:
    - school_year
    - student_id_pseudonimized
    - category_name
    - due_date
    - date_fully_paid

    Algorithm
    ---------
    1. Divide the dataset into two groups: students with single due dates and those with multiple due dates.
        - This is for optimization purposes, as the singular due dates can be calculated via singular matrix operations.
        - While handling multiple due dates requires iterative allocations.
    2. For each group, calculate the amount due, discounts, adjustments, and credit sale transactions.
    3. For students with multiple due dates, allocate discounts and adjustments sequentially across due dates.
    4. Calculate payment allocations into predefined buckets based on days elapsed since due date.
    5. Adds new feature columns for machine learning purposes.
    """
    def __init__(self, df_revenues, df_enrollees, args):
        self.df_revenues = df_revenues.drop(columns=['entry_number'])
        self.df_enrollees = df_enrollees
        self.args = args
        
        self.df_discounts = self._get_discounts(self.df_revenues)
        self.df_adjustments = self._get_adjustments(self.df_revenues)
        self.df_payments_to_receivables = self._get_payments_to_receivables(self.df_revenues)
        self.df_payments_to_all = self._get_payments_to_all(self.df_revenues)

        df_single_due_dates, df_multiple_due_dates = self._get_if_multiple_due_dates(self.df_revenues)
        df_revenues_single, df_revenues_multiple = self._segregate_due_date_types(self.df_revenues, df_single_due_dates, df_multiple_due_dates)
        
        df_credit_sales_single = self._get_credit_sales_single(df_revenues_single)
        df_credit_sales_multiple = self._get_credit_sales_multiple(df_revenues_multiple)
        df_cs = pd.concat([df_credit_sales_single, df_credit_sales_multiple], ignore_index=True)
        print(f"Single due date records: {len(df_credit_sales_single)}")
        print(f"Multiple due date records: {len(df_credit_sales_multiple)}")

        df_cs = self._merge_machine_learning_features(df_cs)

        self.df_cs = df_cs
    
    def _get_discounts(self, df_revenues) -> pd.DataFrame:
        df_disc = df_revenues.query(
            'category_name.str.contains("Discount") and '
            'discount_refund_applied_to != ""'
        ).copy()

        df_disc = (df_disc
            .groupby(['entry_date', 'school_year', 'student_id_pseudonimized', 'category_name', 'discount_refund_applied_to'])
            .sum(numeric_only=True)
            .reset_index()
            .drop(columns=['amount_paid', 'receivables'])
        )
        
        # Rename for compatibility during merging with credit sales
        df_disc = df_disc.drop(columns=['category_name'])
        df_disc.rename(columns={'discount_refund_applied_to': 'category_name',
                                'amount_due': 'amount_discounted'}, inplace=True)
        df_disc = df_disc.groupby(['entry_date', 'school_year', 'student_id_pseudonimized', 'category_name']).sum(numeric_only=True)
        df_disc = df_disc.reset_index()
    
        return df_disc
    
    def _get_adjustments(self, df_revenues) -> pd.DataFrame:
        df_adj = df_revenues.query(
            '`amount_due` < 0 and '
            '`category_name` != "Refund" and '
            'not `category_name`.str.contains("Discount")'
        ).copy()
    
        mask = df_adj['discount_refund_applied_to'].notna() & (df_adj['discount_refund_applied_to'] != '')
        df_adj.loc[mask, 'category_name'] = df_adj.loc[mask, 'discount_refund_applied_to']
    
        df_adj = (
            df_adj[['school_year', 'student_id_pseudonimized', 'category_name', 'amount_due']]
            .rename(columns={'amount_due': 'adjustments'})
            .groupby(['school_year', 'student_id_pseudonimized', 'category_name'], as_index=False)
            .sum()
        )

        return df_adj
    
    def _get_payments_to_receivables(self, df_revenues) -> pd.DataFrame:
        df_p = (df_revenues[['school_year', 'student_id_pseudonimized', 'entry_date',
                        'category_name', 'amount_paid', 'receivables']]
            .query("receivables < 0 and amount_paid.notnull() and amount_paid != 0")
            .rename(columns={'entry_date': 'payment_date'})
            .assign(amount_paid=lambda d: d['amount_paid'].fillna(0))
        )

        return df_p
       
    def _get_payments_to_all(self, df_revenues) -> pd.DataFrame:
        df_p = (df_revenues[['school_year', 'student_id_pseudonimized', 'entry_date',
                        'category_name', 'amount_paid', 'receivables']]
            .query("amount_paid.notnull() and amount_paid != 0")
            .rename(columns={'entry_date': 'payment_date'})
            .assign(amount_paid=lambda d: d['amount_paid'].fillna(0))
        )

        return df_p
    

    
    def _get_if_multiple_due_dates(self, df_revenues) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify students with multiple due dates per school year and per category.

        Parameters
        ----------
        df_revenues : pd.DataFrame
            Input DataFrame containing revenue records. Must include:
            - 'school_year'
            - 'student_id_pseudonimized'
            - 'category_name'
            - 'due_date'

        Returns
        -------
        tuple of pd.DataFrame
            (multiple_due_dates, single_due_date)
        """
        df_revenues = df_revenues[df_revenues['receivables'] > 0 ]

        # Count unique due dates per group
        df_due_dates = (
            df_revenues.groupby(
                ["school_year", "student_id_pseudonimized", "category_name"]
            )["due_date"]
            .nunique()
            .reset_index()
        )

        df_single_due_dates = df_due_dates[df_due_dates["due_date"] == 1].drop(columns="due_date")
        df_multiple_due_dates = df_due_dates[df_due_dates["due_date"] > 1].drop(columns="due_date")

        return df_single_due_dates, df_multiple_due_dates
    
    def _segregate_due_date_types(
            self, df_revenues, df_single_due_date, df_multiple_due_dates
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Segregate the main revenues DataFrame into single and multiple due date records.
        """
        # Merge to get single due date records
        df_revenues_single = pd.merge(
            df_revenues,
            df_single_due_date,
            on=['school_year', 'student_id_pseudonimized', 'category_name'],
            how='inner'
        )

        # Merge to get multiple due date records
        df_revenues_multiple = pd.merge(
            df_revenues,
            df_multiple_due_dates,
            on=['school_year', 'student_id_pseudonimized', 'category_name'],
            how='inner'
        )

        return df_revenues_single, df_revenues_multiple
    
    

    def _get_credit_sales_single(self, df_revenues_single) -> pd.DataFrame:
        """
        Process credit sales for students with single due dates.
        """
        df_ad = self._get_amount_due(df_revenues_single)
        df_cs = self._get_credit_sale_transactions_single(df_ad, self.df_discounts, self.df_adjustments)
        df_dd = self._calculate_due_dates_single(df_revenues_single)
        df_pd = self._calculate_date_fully_paid_single(df_revenues_single, df_dd)

        # Merge due dates and payment dates
        df_cs = pd.merge(df_cs, df_dd, on=['school_year', 'student_id_pseudonimized', 'category_name'], how='left')
        df_cs = pd.merge(df_cs, df_pd, on=['school_year', 'student_id_pseudonimized', 'category_name'], how='left')
        
        return df_cs

    def _get_credit_sales_multiple(self, df_revenues_multiple) -> pd.DataFrame:
        """
        Process credit sales for students with multiple due dates.
        """
        df_ad = self._get_amount_due(df_revenues_multiple)
        df_cs = self._get_credit_sale_transactions_multiple(df_ad, self.df_discounts, self.df_adjustments)
        df_cs = self._merge_latest_payment_dates_multiple(df_revenues_multiple, df_cs)

        return df_cs
    
    

    def _get_amount_due(self, df_revenues) -> pd.DataFrame:
        """
        Calculate and filter amount due records for credit sales.
        """
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
    
        df_ad = df_ad[['entry_date', 'school_year', 'student_id_pseudonimized', 'category_name', 'due_date', 'receivables']]

        return df_ad
    
    ##############################################################
    # Helper Functions to Calculate Credit Sales - Single Due Date
    ##############################################################
    def _get_credit_sale_transactions_single(self, df_ad, df_disc, df_adj) -> pd.DataFrame:
        df_cs = df_ad
        df_cs.rename(columns={'receivables': 'gross_receivables'}, inplace=True)
        df_cs = df_cs.groupby(['school_year', 'student_id_pseudonimized', 'category_name']).sum(numeric_only=True)
        df_disc = df_disc.groupby(['school_year', 'student_id_pseudonimized', 'category_name']).sum(numeric_only=True)
        
        df_cs = pd.merge(df_cs, df_disc, on=['school_year', 'student_id_pseudonimized', 'category_name'], how='left')
        df_cs = pd.merge(df_cs, df_adj, on=['school_year', 'student_id_pseudonimized', 'category_name'], how='left')
        
        # Fill n/a with 0 the calculate adjusted credit sale amount
        df_cs[['amount_discounted', 'adjustments']] = df_cs[['amount_discounted', 'adjustments']].fillna(0)
        df_cs['credit_sale_amount'] = df_cs['gross_receivables']\
            + df_cs['amount_discounted'] \
            + df_cs['adjustments']
        
        # Filter with zero credit sale
        df_cs = df_cs[df_cs['credit_sale_amount'] != 0]
    
        return df_cs

    def _calculate_due_dates_single(self, df_revenues) -> pd.DataFrame:
        df_dd = df_revenues[['school_year', 'student_id_pseudonimized', 'category_name', 'amount_due', 'due_date', 'receivables']]
        df_dd = df_dd[(df_dd['amount_due'] != 0) & (df_dd['receivables'] != 0)]
        df_dd = df_dd.groupby(['school_year', 'student_id_pseudonimized', 'category_name']).min()
        df_dd.reset_index(inplace=True)
        df_dd = df_dd.drop(columns=['amount_due', 'receivables'])
        
        return df_dd

    def _calculate_date_fully_paid_single(self, df_revenues, df_dd) -> pd.DataFrame:
        # Merge with invoice data
        df_p = pd.merge(
            df_dd,
            self.df_payments_to_receivables,
            on=['school_year', 'student_id_pseudonimized', 'category_name'],
            how='left'
        )
        
        # Clean up
        df_p = df_p.drop(columns=['receivables'])
        df_p = df_p.dropna(subset=['amount_paid'])
        df_p = df_p[df_p['amount_paid'] != 0]
        
        # Get the last payment date per invoice
        df_p = (
            df_p.groupby(['school_year', 'student_id_pseudonimized', 'category_name'])
                .agg(date_fully_paid=('payment_date', 'max'))
                .reset_index()
        )

        return df_p


    #################################################################
    # Helper Functions to Calculate Credit Sales - Multiple Due Dates
    #################################################################
    def _get_credit_sale_transactions_multiple(self, df_ad, df_disc, df_adj) -> pd.DataFrame:
        df_cs = df_ad.copy()
        df_cs.rename(columns={'receivables': 'gross_receivables'}, inplace=True)

        # Sort receivables chronologically
        df_cs.sort_values(
            by=['school_year', 'student_id_pseudonimized', 'category_name', 'due_date'],
            inplace=True
        )

        # --- Prepare discounts ---
        # Ensure 'amount_discounted' column exists
        if 'amount_discounted' not in df_disc.columns:
            df_disc['amount_discounted'] = 0.0

        # Pre-group discounts for faster access
        disc_groups = {
            key: subdf.sort_values('entry_date').to_numpy()
            for key, subdf in df_disc.groupby(['school_year','student_id_pseudonimized','category_name'])
        }

        # Precompute discount column index once
        disc_amount_idx = df_disc.columns.get_loc('amount_discounted')

        # Convert adjustments to dict for O(1) lookup
        adj_dict = dict(
            zip(
                zip(df_adj['school_year'], df_adj['student_id_pseudonimized'], df_adj['category_name']),
                df_adj['adjustments']
            )
        )

        # Parallel processing of groups
        grouped = df_cs.groupby(['school_year','student_id_pseudonimized','category_name'])
        args = ((g, disc_groups, adj_dict, disc_amount_idx) for _, g in grouped)

        with Pool(processes=cpu_count()) as pool:
            results = pool.map(_allocate_discount_and_adjustments, args)

        if results:
            df_cs = pd.concat(results).reset_index(drop=True)
            df_cs = df_cs[df_cs['credit_sale_amount'] != 0]

        return df_cs
    
    def _merge_latest_payment_dates_multiple(self, df_revenues_multiple: pd.DataFrame, df_cs: pd.DataFrame) -> pd.DataFrame:
        # --- Prepare payment records ---
        df_p = (
            df_revenues_multiple[['school_year', 'student_id_pseudonimized', 'entry_date',
                         'category_name', 'amount_paid', 'receivables']]
            .query("receivables < 0 and amount_paid.notnull() and amount_paid != 0")
            .rename(columns={'entry_date': 'payment_date'})
            .assign(amount_paid=lambda d: d['amount_paid'].fillna(0))
        )


        # MultiIndex for grouping
        df_p.set_index(['school_year', 'student_id_pseudonimized', 'category_name'], inplace=True)

        # --- Build tasks for multiprocessing ---
        tasks = []
        for keys, payments in df_p.groupby(level=[0, 1, 2]):
            receivables = df_cs[
                (df_cs['school_year'] == keys[0]) &
                (df_cs['student_id_pseudonimized'] == keys[1]) &
                (df_cs['category_name'] == keys[2])
            ].sort_values(by='due_date').copy()

            # Keep original index to merge later to support multiple due_date's
            receivables['cs_index'] = receivables.index
            tasks.append((receivables, payments))

        # --- Run in parallel ---
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(_allocate_date_fully_paid_sequential, tasks)

        # Concatenate results and merge back
        df_fully_paid_dates = pd.concat(results, ignore_index=True)

        df_cs = df_cs.merge(
            df_fully_paid_dates[['cs_index', 'date_fully_paid']],
            left_index=True,
            right_on='cs_index',
            how='left'
        ).drop(columns=['cs_index', 'entry_date'])

        return df_cs

    def _merge_last_payment_date(self, df_cs, df_p) -> pd.DataFrame:
        """
        For each receivable, get the latest payment date before its due_date,
        restricted to the same student_id_pseudonimized, but only from
        payments in a different category_name.
        """
        # Sort for clarity
        df_p = df_p.sort_values('payment_date').copy()
        df_cs = df_cs.sort_values('due_date').copy()

        # --- Vectorized join with student_id_pseudonimized ---
        merged = df_cs[['student_id_pseudonimized', 'category_name', 'due_date']].merge(
            df_p[['student_id_pseudonimized', 'category_name', 'payment_date']],
            on='student_id_pseudonimized',
            how='inner'
        )

        # Exclude same-category matches
        merged = merged.loc[merged['category_name_x'] != merged['category_name_y']]

        # Keep only payments before due_date
        merged = merged.loc[merged['payment_date'] < merged['due_date']]

        # --- Groupby aggregation ---
        result = (
            merged.groupby(['student_id_pseudonimized', 'due_date'])['payment_date']
                .max()
                .reset_index()
                .rename(columns={'payment_date': 'last_payment_date'})
        )

        # Merge back to original receivables
        result = df_cs.merge(result, on=['student_id_pseudonimized', 'due_date'], how='left')

        return result


    ###########################################################
    # Helper Functions to Extract Features for Machine Learning
    ###########################################################
    def _merge_machine_learning_features(self, df_cs) -> pd.DataFrame:
        # Days to pay features used for time-series features
        df_cs = self._merge_dtp(df_cs, dtp_n=4)

        # Averages and weighted averages
        df_cs['dtp_avg'] = df_cs[['dtp_1', 'dtp_2', 'dtp_3', 'dtp_4']].mean(axis=1)
        weights = [0.4, 0.3, 0.2, 0.1]
        df_cs['dtp_wavg'] = df_cs[['dtp_1', 'dtp_2', 'dtp_3', 'dtp_4']].mul(weights).sum(axis=1)

        df_cs = self._merge_due_date_prev(df_cs, 2)

        # Time-normalized trends
        df_cs['dtp_2_trend'] = self._calculate_dtp_trend(df_cs, lag=1)
        df_cs['dtp_3_trend'] = self._calculate_dtp_trend(df_cs, lag=2)

        # Days since last payment
        df_cs = self._merge_last_payment_date(df_cs, self.df_payments_to_all)
        df_cs['days_since_last_payment'] = (
            df_cs['due_date'] - df_cs['last_payment_date']
        ).dt.days.astype("Int64").fillna(-1)

        # Balance calculations
        df_cs = self._merge_amount_due_cum_sum(df_cs, self.df_revenues)
        df_cs = self._merge_amount_paid_cum_sum(df_cs, self.df_revenues)
        df_cs['opening_balance'] = (
            df_cs['amount_due_cumsum'] - df_cs['amount_paid_cumsum']
        ).clip(lower=0)

        # Merge enrollee info
        df_cs = df_cs.merge(
            self.df_enrollees[['school_year', 'student_id_pseudonimized', 'plan_type']],
            on=['school_year', 'student_id_pseudonimized'],
            how='left'
        )

        # Encoding and brackets
        df_cs = self._apply_one_hot_encoding(df_cs)
        df_cs = self._merge_dtp_bracket(df_cs)
        df_cs = self._merge_censor(df_cs)
        df_cs = self._calculate_dtp_for_censored_invoices(df_cs)

        return df_cs

    def _merge_dtp(self, df_cs, dtp_n) -> pd.DataFrame:
        """
        Extract the number of days between the current invoice due date and the
        Nth previous issued invoice (where N = no_of_invoices_back).
        Keeps date_fully_paid blank if unpaid, but still calculates dtp_# using shift.
        """

        # Ensure sorted by student_id, then due_date (across all years)
        df_cs = df_cs.sort_values(
            ["student_id_pseudonimized", "due_date"]
        )

        # Compute days_elapsed_until_fully_paid
        df_cs["days_elapsed_until_fully_paid"] = (
            df_cs["date_fully_paid"] - df_cs["due_date"]
        ).dt.days.astype("Int64")  # Int64 stays <NA> if unpaid

        # Use shift to get previous N values (across all years for the same student)
        for n in range(1, dtp_n+1):
            df_cs[f"dtp_{n}"] = (
                df_cs.groupby("student_id_pseudonimized")["days_elapsed_until_fully_paid"]
                .shift(n)
                .astype("Int64")
            )

        return df_cs

    def _merge_due_date_prev(self, df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
        """
        Adds 'due_date_prev_k' columns per student_id_pseudonimized, representing the
        previous k-th invoice's due_date for that student.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing 'student_id_pseudonimized' and 'due_date'.
        n : int, optional (default=1)
            Number of previous entries to retrieve. If n=1, only the immediate
            previous due_date is added. If n>1, multiple columns are added.

        Returns
        -------
        pd.DataFrame
            Dataframe with additional 'due_date_prev_k' columns.
        """
        # Ensure due_date is datetime
        df['due_date'] = pd.to_datetime(df['due_date'])

        # Sort by student and due_date
        df = df.sort_values(['student_id_pseudonimized', 'due_date'])

        # Generate shifted columns dynamically
        for k in range(1, n + 1):
            df[f'due_date_prev_{k}'] = (
                df.groupby('student_id_pseudonimized')['due_date']
                .shift(k)
            )

        return df

    def _calculate_dtp_trend(self, df: pd.DataFrame, lag: int) -> pd.Series:
        """
        Calculate normalized DTP trend between two invoices based on due_date differences.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing DTP and due_date columns.
        lag : int
            How many invoices back to compare (e.g., 1 for dtp_2 vs dtp_1).

        Returns
        -------
        pd.Series
            Normalized trend values (rate of change per day).
        """
        # Current column is always dtp_{lag+1}, previous is dtp_{1}
        col_current = f'dtp_{lag+1}'
        col_previous = 'dtp_1'

        date_current = df['due_date']
        column_prev = f'due_date_prev_{lag}'
        date_previous = df[column_prev]

        days_diff = (date_current - date_previous).dt.days
        trend = (df[col_current] - df[col_previous]) / days_diff.replace(0, pd.NA)

        return trend.fillna(0)
    
    def _merge_amount_due_cum_sum(self, df_cs: pd.DataFrame, df_revenues: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the cumulative sum of df_revenues['amount_due'] for each student in df_cs
        where df_revenues['due_date'] <= df_cs['due_date'].
        """
        # Sort revenues by due_date for cumulative sum
        df_revenues = df_revenues.sort_values(['student_id_pseudonimized', 'due_date'])

        # Compute cumulative sum per student
        df_revenues['amount_due_cumsum'] = df_revenues.groupby('student_id_pseudonimized')['amount_due'].cumsum()

        # For each df_cs row, we want the latest cumulative sum where revenue.due_date <= cs.due_date
        # Merge on student_id_pseudonimized, then filter by due_date condition
        merged = pd.merge(df_cs, df_revenues, on='student_id_pseudonimized', suffixes=('_cs', '_rev'))

        # Keep only rows where revenue due_date <= cs due_date
        merged = merged[merged['due_date_rev'] <= merged['due_date_cs']]

        # For each cs row, take the max cumulative sum (latest revenue before cs.due_date)
        result = (
            merged.groupby(['student_id_pseudonimized', 'due_date_cs'], as_index=False)['amount_due_cumsum']
            .max()
        )

        # Merge back into df_cs
        df_cs = pd.merge(df_cs, result, left_on=['student_id_pseudonimized', 'due_date'], right_on=['student_id_pseudonimized', 'due_date_cs'], how='left')

        # Drop helper column
        df_cs = df_cs.drop(columns=['due_date_cs'])

        return df_cs

    def _merge_amount_paid_cum_sum(self, df_cs: pd.DataFrame, df_revenues: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the cumulative sum of df_revenues['amount_paid'] for each student in df_cs
        where df_revenues['due_date'] <= df_cs['due_date'].
        """
        # Sort revenues by entry_date for cumulative sum
        df_revenues = df_revenues.sort_values(['student_id_pseudonimized', 'entry_date'])

        # Compute cumulative sum per student
        df_revenues['amount_paid_cumsum'] = df_revenues.groupby('student_id_pseudonimized')['amount_paid'].cumsum()

        # For each df_cs row, we want the latest cumulative sum where revenue.due_date <= cs.due_date
        # Merge on student_id_pseudonimized, then filter by due_date condition
        merged = pd.merge(df_cs, df_revenues, on='student_id_pseudonimized', suffixes=('_cs', '_rev'))

        # Keep only rows where revenue entry_date <= cs due_date
        merged = merged[merged['entry_date'] <= merged['due_date_cs']]

        # For each cs row, take the max cumulative sum (latest revenue before cs.due_date)
        result = (
            merged.groupby(['student_id_pseudonimized', 'due_date_cs'], as_index=False)['amount_paid_cumsum']
            .max()
        )

        # Merge back into df_cs
        df_cs = pd.merge(df_cs, result, left_on=['student_id_pseudonimized', 'due_date'], right_on=['student_id_pseudonimized', 'due_date_cs'], how='left')

        # Drop helper column
        df_cs = df_cs.drop(columns=['due_date_cs'])

        return df_cs

    def _apply_one_hot_encoding(self, df_cs: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical features while keeping original columns.
        """
        categorical_features = ['plan_type']

        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(df_cs[categorical_features])

        df_cs_encoded = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(categorical_features),
            index=df_cs.index
        )

        # Concatenate original df_cs with encoded features
        df_cs = pd.concat([df_cs, df_cs_encoded], axis=1)
        df_cs.drop(columns=categorical_features, inplace=True)

        return df_cs
    
    def _merge_dtp_bracket(self, df_cs):
        # Define bracket rules in a dictionary
        bracket_rules = {
            "on_time": lambda x: x <= 0,
            "30_days": lambda x: 0 < x <= 30,  # 1-30 days
            "60_days": lambda x: 30 < x <= 60, # 31-60 days
            "90_days": lambda x: x > 60        # 61-90 days
        }

        # Apply rules to create the new column
        def assign_bracket(x):
            for label, rule in bracket_rules.items():
                if rule(x):
                    return label
            return None  # default if no condition matches

        df_cs['dtp_bracket'] = df_cs['days_elapsed_until_fully_paid'].apply(assign_bracket)

        return df_cs
    
    def _merge_censor(self, df_cs):
        """
        Adds a new column that is used in survival analysis.
        censor (E) = 0: The payment still has not happened yet during the observation end date
        censor (E) = 1: The exact date of full payment happened during the observation end date
        """
        observation_end_date = df_cs['due_date'].max()
        df_cs['censor'] = (df_cs['date_fully_paid'] <= observation_end_date).astype(int)

        return df_cs
    
    def _calculate_dtp_for_censored_invoices(self, df_cs):
        """
        Calculate days_elapsed_until_fully_paid for censored data (censor = 0).
        Result is stored as integer days for all rows.
        """
        default_observation_end_date = datetime.today()
        observation_end_date = getattr(self.args, "observation_end", default_observation_end_date)

        # Keep records before observation_end_date regardless of payment,
        # plus those after observation_end_date only if they are fully paid
        df_cs = df_cs[
            (df_cs['due_date'] <= observation_end_date) |
            ((df_cs['due_date'] > observation_end_date) & df_cs['date_fully_paid'].notna())
        ]

        # Update only censored rows
        df_cs.loc[df_cs['censor'] == 0, 'days_elapsed_until_fully_paid'] = (
            (observation_end_date - df_cs.loc[df_cs['censor'] == 0, 'due_date']).dt.days
        )

        # Convert entire column to integer safely
        df_cs['days_elapsed_until_fully_paid'] = (
            df_cs['days_elapsed_until_fully_paid']
            .infer_objects(copy=False)
            .fillna(0)
            .astype(int)
        )

        return df_cs
    
        
    def show_data(self):
        return self.df_cs

# --- helper functions that must be top-level for multiprocessing ---
def _allocate_discount_and_adjustments(args) -> pd.DataFrame:
    """
    Allocate discounts and adjustments sequentially for a group of receivables.
    """
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

def _allocate_date_fully_paid_sequential(args) -> pd.DataFrame:
    """
    Sequentially allocate payments across receivables and record
    the payment date that fully settled each receivable.
    """
    receivables, payments = args

    # Ensure datetime types
    payments['payment_date'] = pd.to_datetime(payments['payment_date'], errors='coerce')
    receivables['due_date'] = pd.to_datetime(receivables['due_date'], errors='coerce')

    # Sort both
    payments = payments.sort_values('payment_date').copy()
    receivables = receivables.sort_values('due_date').copy()

    # Track balances
    receivables['remaining'] = receivables['credit_sale_amount']
    receivables['date_fully_paid'] = pd.NaT

    # Iterate payments sequentially
    for _, pay in payments.iterrows():
        amt = pay['amount_paid']
        pay_date = pay['payment_date']

        for i in receivables.index:
            if amt <= 0:
                break

            if receivables.at[i, 'remaining'] > 0:
                apply_amt = min(amt, receivables.at[i, 'remaining'])
                receivables.at[i, 'remaining'] -= apply_amt
                amt -= apply_amt

                # If this payment fully cleared the receivable, mark its date
                if receivables.at[i, 'remaining'] == 0:
                    receivables.at[i, 'date_fully_paid'] = pay_date

    # Return with original index preserved
    result = receivables[['cs_index', 'date_fully_paid']].copy()

    return result