import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from sklearn.preprocessing import OneHotEncoder


pd.set_option('future.no_silent_downcasting', True)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers — must be top-level for ThreadPool.map pickling
# ─────────────────────────────────────────────────────────────────────────────

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
        amount_discounted=disc_applied,
        adjustments=adj_applied,
        credit_sale_amount=gross + disc_applied + adj_applied
    )


def _allocate_date_fully_paid_sequential(args) -> pd.DataFrame:
    """
    Sequentially allocate payments across receivables and record
    the payment date that fully settled each receivable.
    """
    receivables, payments = args

    payments['payment_date'] = pd.to_datetime(payments['payment_date'], errors='coerce')
    receivables['due_date'] = pd.to_datetime(receivables['due_date'], errors='coerce')

    payments = payments.sort_values('payment_date').copy()
    receivables = receivables.sort_values('due_date').copy()

    receivables['remaining'] = receivables['credit_sale_amount']
    receivables['date_fully_paid'] = pd.NaT

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
                if receivables.at[i, 'remaining'] == 0:
                    receivables.at[i, 'date_fully_paid'] = pay_date

    return receivables[['cs_index', 'date_fully_paid']].copy()


def _allocate_payment_amounts_sequential(args) -> pd.DataFrame:
    """
    Sequentially allocate payments across receivables (earliest due date first)
    and record the total payment amount applied to each receivable, along with
    any portion received on or before the due date (prepayments).
    """
    receivables, payments = args

    payments['payment_date'] = pd.to_datetime(payments['payment_date'], errors='coerce')
    receivables['due_date'] = pd.to_datetime(receivables['due_date'], errors='coerce')

    payments = payments.sort_values('payment_date').copy()
    receivables = receivables.sort_values('due_date').copy()

    receivables['remaining'] = receivables['credit_sale_amount']
    receivables['prepayments'] = 0.0
    receivables['total_payments'] = 0.0

    for _, pay in payments.iterrows():
        amt = pay['amount_paid']
        pay_date = pay['payment_date']

        for i in receivables.index:
            if amt <= 0:
                break
            if receivables.at[i, 'remaining'] > 0:
                apply_amt = min(amt, receivables.at[i, 'remaining'])
                receivables.at[i, 'remaining'] -= apply_amt
                receivables.at[i, 'total_payments'] += apply_amt
                amt -= apply_amt
                if pd.notnull(pay_date) and pd.notnull(receivables.at[i, 'due_date']):
                    if pay_date <= receivables.at[i, 'due_date']:
                        receivables.at[i, 'prepayments'] += apply_amt

    return receivables[['cs_index', 'prepayments', 'total_payments']].copy()


# ─────────────────────────────────────────────────────────────────────────────
# InvoiceBuilder
# ─────────────────────────────────────────────────────────────────────────────

class InvoiceBuilder:
    """
    Constructs a clean invoice-level DataFrame from raw revenue records.

    Handles all financial logic: discount allocation, adjustment allocation,
    payment date resolution, and optional payment-amount bucketing. Produces
    one row per receivable with columns for amounts, due dates, and payment dates.
    Has no knowledge of ML features or post-processing.

    Parameters
    ----------
    df_revenues : pd.DataFrame
        Revenue records with ``entry_number`` already dropped.
    calculate_payment_amounts : bool, default False
        If True, adds ``prepayments``, ``total_payments``, ``adjusted_credit_amount``,
        and ``net_receivables`` columns.
    """

    def __init__(self, df_revenues: pd.DataFrame, calculate_payment_amounts: bool = False):
        self.df_revenues = df_revenues
        self.calculate_payment_amounts = calculate_payment_amounts

        self.df_discounts = self._get_discounts(df_revenues)
        self.df_adjustments = self._get_adjustments(df_revenues)
        self.df_payments_to_receivables = self._get_payments_to_receivables(df_revenues)
        self.df_payments_to_all = self._get_payments_to_all(df_revenues)

    def build(self) -> pd.DataFrame:
        """
        Build and return the invoice-level DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per receivable. Columns include school_year,
            student_id_pseudonimized, category_name, due_date,
            date_fully_paid, gross_receivables, amount_discounted,
            adjustments, and credit_sale_amount.
        """
        df_single_due_dates, df_multiple_due_dates = self._get_if_multiple_due_dates(
            self.df_revenues
        )
        df_revenues_single, df_revenues_multiple = self._segregate_due_date_types(
            self.df_revenues, df_single_due_dates, df_multiple_due_dates
        )

        df_cs_single = self._get_credit_sales_single(df_revenues_single)
        df_cs_multiple = self._get_credit_sales_multiple(df_revenues_multiple)
        df_cs = pd.concat([df_cs_single, df_cs_multiple], ignore_index=True)

        print(f"Single due date records:   {len(df_cs_single)}")
        print(f"Multiple due date records: {len(df_cs_multiple)}")

        if self.calculate_payment_amounts:
            df_cs[['prepayments', 'total_payments']] = (
                df_cs[['prepayments', 'total_payments']].fillna(0)
            )
            df_cs['adjusted_credit_amount'] = (
                df_cs['credit_sale_amount'] - df_cs['prepayments']
            )
            df_cs['net_receivables'] = (
                df_cs['credit_sale_amount'] - df_cs['total_payments']
            )

        return df_cs

    # ── Raw extraction ────────────────────────────────────────────────────────

    def _get_discounts(self, df_revenues: pd.DataFrame) -> pd.DataFrame:
        df_disc = df_revenues.query(
            'category_name.str.contains("Discount") and '
            'discount_refund_applied_to != ""'
        ).copy()

        df_disc = (
            df_disc
            .groupby(['entry_date', 'school_year', 'student_id_pseudonimized',
                      'category_name', 'discount_refund_applied_to'])
            .sum(numeric_only=True)
            .reset_index()
            .drop(columns=['amount_paid', 'receivables'])
        )

        df_disc = df_disc.drop(columns=['category_name'])
        df_disc.rename(
            columns={'discount_refund_applied_to': 'category_name',
                     'amount_due': 'amount_discounted'},
            inplace=True
        )
        df_disc = (
            df_disc
            .groupby(['entry_date', 'school_year', 'student_id_pseudonimized', 'category_name'])
            .sum(numeric_only=True)
            .reset_index()
        )

        return df_disc

    def _get_adjustments(self, df_revenues: pd.DataFrame) -> pd.DataFrame:
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

    def _get_payments_to_receivables(self, df_revenues: pd.DataFrame) -> pd.DataFrame:
        df_p = (
            df_revenues[['school_year', 'student_id_pseudonimized', 'entry_date',
                         'category_name', 'amount_paid', 'receivables']]
            .query("receivables < 0 and amount_paid.notnull() and amount_paid != 0")
            .rename(columns={'entry_date': 'payment_date'})
            .assign(amount_paid=lambda d: d['amount_paid'].fillna(0))
        )
        return df_p

    def _get_payments_to_all(self, df_revenues: pd.DataFrame) -> pd.DataFrame:
        df_p = (
            df_revenues[['school_year', 'student_id_pseudonimized', 'entry_date',
                         'category_name', 'amount_paid', 'receivables']]
            .query("amount_paid.notnull() and amount_paid != 0")
            .rename(columns={'entry_date': 'payment_date'})
            .assign(amount_paid=lambda d: d['amount_paid'].fillna(0))
        )
        return df_p

    # ── Due-date segmentation ─────────────────────────────────────────────────

    def _get_if_multiple_due_dates(
        self, df_revenues: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_revenues = df_revenues[df_revenues['receivables'] > 0]
        df_due_dates = (
            df_revenues
            .groupby(['school_year', 'student_id_pseudonimized', 'category_name'])['due_date']
            .nunique()
            .reset_index()
        )
        df_single = df_due_dates[df_due_dates['due_date'] == 1].drop(columns='due_date')
        df_multiple = df_due_dates[df_due_dates['due_date'] > 1].drop(columns='due_date')
        return df_single, df_multiple

    def _segregate_due_date_types(
        self,
        df_revenues: pd.DataFrame,
        df_single_due_date: pd.DataFrame,
        df_multiple_due_dates: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_revenues_single = pd.merge(
            df_revenues, df_single_due_date,
            on=['school_year', 'student_id_pseudonimized', 'category_name'],
            how='inner'
        )
        df_revenues_multiple = pd.merge(
            df_revenues, df_multiple_due_dates,
            on=['school_year', 'student_id_pseudonimized', 'category_name'],
            how='inner'
        )
        return df_revenues_single, df_revenues_multiple

    # ── Single due-date path ──────────────────────────────────────────────────

    def _get_credit_sales_single(self, df_revenues_single: pd.DataFrame) -> pd.DataFrame:
        df_ad = self._get_amount_due(df_revenues_single)
        df_cs = self._get_credit_sale_transactions_single(
            df_ad, self.df_discounts, self.df_adjustments
        )
        df_dd = self._calculate_due_dates_single(df_revenues_single)
        df_pd = self._calculate_date_fully_paid_single(df_revenues_single, df_dd)

        df_cs = pd.merge(df_cs, df_dd, on=['school_year', 'student_id_pseudonimized', 'category_name'], how='left')
        df_cs = pd.merge(df_cs, df_pd, on=['school_year', 'student_id_pseudonimized', 'category_name'], how='left')

        if self.calculate_payment_amounts:
            df_pa = self._calculate_payment_amounts_single(df_revenues_single, df_dd)
            df_cs = pd.merge(df_cs, df_pa, on=['school_year', 'student_id_pseudonimized', 'category_name'], how='left')

        return df_cs

    def _get_amount_due(self, df_revenues: pd.DataFrame) -> pd.DataFrame:
        df_has_amount_due = df_revenues.groupby(
            ['school_year', 'student_id_pseudonimized', 'category_name']
        ).sum(numeric_only=True)
        df_has_amount_due = df_has_amount_due[df_has_amount_due['amount_due'] == 0]
        df_has_amount_due = df_has_amount_due.reset_index().drop(columns=['amount_due', 'amount_paid'])

        common_rows = pd.merge(
            df_revenues, df_has_amount_due,
            on=['school_year', 'student_id_pseudonimized', 'category_name'],
            how='inner'
        )
        df_ad = df_revenues[
            ~df_revenues.set_index(['school_year', 'student_id_pseudonimized', 'category_name'])
            .index.isin(common_rows.set_index(['school_year', 'student_id_pseudonimized', 'category_name']).index)
        ]
        df_ad.reset_index()

        df_ad = df_ad.query(
            '`amount_due` != 0 and `receivables` > 0 and `amount_paid` >= 0 and '
            '`category_name` != "Refund" and `category_name` != "Overpayment" and '
            'not `category_name`.str.contains("Discount")'
        )
        df_ad = df_ad[['entry_date', 'school_year', 'student_id_pseudonimized', 'category_name', 'due_date', 'receivables']]
        return df_ad

    def _get_credit_sale_transactions_single(
        self,
        df_ad: pd.DataFrame,
        df_disc: pd.DataFrame,
        df_adj: pd.DataFrame,
    ) -> pd.DataFrame:
        df_cs = df_ad
        df_cs.rename(columns={'receivables': 'gross_receivables'}, inplace=True)
        df_cs = df_cs.groupby(['school_year', 'student_id_pseudonimized', 'category_name']).sum(numeric_only=True)
        df_disc = df_disc.groupby(['school_year', 'student_id_pseudonimized', 'category_name']).sum(numeric_only=True)

        df_cs = pd.merge(df_cs, df_disc, on=['school_year', 'student_id_pseudonimized', 'category_name'], how='left')
        df_cs = pd.merge(df_cs, df_adj, on=['school_year', 'student_id_pseudonimized', 'category_name'], how='left')

        df_cs[['amount_discounted', 'adjustments']] = df_cs[['amount_discounted', 'adjustments']].fillna(0)
        df_cs['credit_sale_amount'] = (
            df_cs['gross_receivables'] + df_cs['amount_discounted'] + df_cs['adjustments']
        )
        df_cs = df_cs[df_cs['credit_sale_amount'] != 0]
        return df_cs

    def _calculate_due_dates_single(self, df_revenues: pd.DataFrame) -> pd.DataFrame:
        df_dd = df_revenues[['school_year', 'student_id_pseudonimized', 'category_name', 'amount_due', 'due_date', 'receivables']]
        df_dd = df_dd[(df_dd['amount_due'] != 0) & (df_dd['receivables'] != 0)]
        df_dd = df_dd.groupby(['school_year', 'student_id_pseudonimized', 'category_name']).min()
        df_dd.reset_index(inplace=True)
        df_dd = df_dd.drop(columns=['amount_due', 'receivables'])
        return df_dd

    def _calculate_date_fully_paid_single(
        self, df_revenues: pd.DataFrame, df_dd: pd.DataFrame
    ) -> pd.DataFrame:
        df_p = pd.merge(
            df_dd, self.df_payments_to_receivables,
            on=['school_year', 'student_id_pseudonimized', 'category_name'],
            how='left'
        )
        df_p = df_p.drop(columns=['receivables'])
        df_p = df_p.dropna(subset=['amount_paid'])
        df_p = df_p[df_p['amount_paid'] != 0]
        df_p = (
            df_p.groupby(['school_year', 'student_id_pseudonimized', 'category_name'])
            .agg(date_fully_paid=('payment_date', 'max'))
            .reset_index()
        )
        return df_p

    def _calculate_payment_amounts_single(
        self, df_revenues: pd.DataFrame, df_dd: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate total payment amounts for single due date invoices without bucketing.

        For each invoice, sums all payments received and splits them into:
        - prepayments : payments received on or before the due date
        - total_payments : all payments received regardless of timing

        Parameters
        ----------
        df_revenues : pd.DataFrame
            Revenue records for single due date students.
        df_dd : pd.DataFrame
            Due dates per student/category (from _calculate_due_dates_single).

        Returns
        -------
        pd.DataFrame
            Columns: school_year, student_id_pseudonimized, category_name,
                     prepayments, total_payments
        """
        df_p = df_revenues[['school_year', 'student_id_pseudonimized', 'entry_date',
                             'category_name', 'amount_paid', 'receivables']]
        df_p = df_p[df_p['receivables'] < 0]
        df_p = pd.merge(
            df_dd, df_p,
            on=['school_year', 'student_id_pseudonimized', 'category_name'],
            how='left'
        )
        df_p = df_p.drop(columns=['receivables'])
        df_p.rename(columns={'entry_date': 'payment_date'}, inplace=True)
        df_p = df_p.dropna(subset=['amount_paid'])
        df_p = df_p[df_p['amount_paid'] != 0]

        df_p['days_elapsed'] = (df_p['payment_date'] - df_p['due_date']).dt.days
        df_p['prepayments'] = df_p.apply(
            lambda r: r['amount_paid'] if r['days_elapsed'] <= 0 else 0, axis=1
        )
        df_p = df_p.groupby(['school_year', 'student_id_pseudonimized', 'category_name']).agg(
            prepayments=('prepayments', 'sum'),
            total_payments=('amount_paid', 'sum')
        ).reset_index()
        return df_p

    # ── Multiple due-date path ────────────────────────────────────────────────

    def _get_credit_sales_multiple(self, df_revenues_multiple: pd.DataFrame) -> pd.DataFrame:
        df_ad = self._get_amount_due(df_revenues_multiple)
        df_cs = self._get_credit_sale_transactions_multiple(
            df_ad, self.df_discounts, self.df_adjustments
        )
        df_cs = self._merge_latest_payment_dates_multiple(df_revenues_multiple, df_cs)

        if self.calculate_payment_amounts:
            df_cs = self._merge_payment_amounts_multiple(df_revenues_multiple, df_cs)

        return df_cs

    def _get_credit_sale_transactions_multiple(
        self,
        df_ad: pd.DataFrame,
        df_disc: pd.DataFrame,
        df_adj: pd.DataFrame,
    ) -> pd.DataFrame:
        df_cs = df_ad.copy()
        df_cs.rename(columns={'receivables': 'gross_receivables'}, inplace=True)
        df_cs.sort_values(
            by=['school_year', 'student_id_pseudonimized', 'category_name', 'due_date'],
            inplace=True
        )

        if 'amount_discounted' not in df_disc.columns:
            df_disc['amount_discounted'] = 0.0

        disc_groups = {
            key: subdf.sort_values('entry_date').to_numpy()
            for key, subdf in df_disc.groupby(['school_year', 'student_id_pseudonimized', 'category_name'])
        }
        disc_amount_idx = df_disc.columns.get_loc('amount_discounted')
        adj_dict = dict(
            zip(
                zip(df_adj['school_year'], df_adj['student_id_pseudonimized'], df_adj['category_name']),
                df_adj['adjustments']
            )
        )

        grouped = df_cs.groupby(['school_year', 'student_id_pseudonimized', 'category_name'])
        args = ((g, disc_groups, adj_dict, disc_amount_idx) for _, g in grouped)

        with ThreadPool(processes=cpu_count()) as pool:
            results = pool.map(_allocate_discount_and_adjustments, args)

        if results:
            df_cs = pd.concat(results).reset_index(drop=True)
            df_cs = df_cs[df_cs['credit_sale_amount'] != 0]

        return df_cs

    def _merge_latest_payment_dates_multiple(
        self, df_revenues_multiple: pd.DataFrame, df_cs: pd.DataFrame
    ) -> pd.DataFrame:
        df_p = (
            df_revenues_multiple[['school_year', 'student_id_pseudonimized', 'entry_date',
                                   'category_name', 'amount_paid', 'receivables']]
            .query("receivables < 0 and amount_paid.notnull() and amount_paid != 0")
            .rename(columns={'entry_date': 'payment_date'})
            .assign(amount_paid=lambda d: d['amount_paid'].fillna(0))
        )
        df_p.set_index(['school_year', 'student_id_pseudonimized', 'category_name'], inplace=True)

        tasks = []
        for keys, payments in df_p.groupby(level=[0, 1, 2]):
            receivables = df_cs[
                (df_cs['school_year'] == keys[0]) &
                (df_cs['student_id_pseudonimized'] == keys[1]) &
                (df_cs['category_name'] == keys[2])
            ].sort_values(by='due_date').copy()
            receivables['cs_index'] = receivables.index
            tasks.append((receivables, payments))

        with ThreadPool(processes=cpu_count()) as pool:
            results = pool.map(_allocate_date_fully_paid_sequential, tasks)

        df_fully_paid_dates = pd.concat(results, ignore_index=True)
        df_cs = df_cs.merge(
            df_fully_paid_dates[['cs_index', 'date_fully_paid']],
            left_index=True,
            right_on='cs_index',
            how='left'
        ).drop(columns=['cs_index', 'entry_date'])

        return df_cs

    def _merge_payment_amounts_multiple(
        self, df_revenues_multiple: pd.DataFrame, df_cs: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate total payment amounts for multiple due date invoices without bucketing.

        Payments are allocated sequentially across due dates (earliest first).
        For each receivable, records:
        - prepayments : portion of payments received on or before its due date
        - total_payments : total payments applied to this receivable

        Parameters
        ----------
        df_revenues_multiple : pd.DataFrame
            Revenue records for multiple due date students.
        df_cs : pd.DataFrame
            Credit sales DataFrame with credit_sale_amount and due_date per row.

        Returns
        -------
        pd.DataFrame
            df_cs with added 'prepayments' and 'total_payments' columns.
        """
        df_p = (
            df_revenues_multiple[['school_year', 'student_id_pseudonimized', 'entry_date',
                                   'category_name', 'amount_paid', 'receivables']]
            .query("receivables < 0 and amount_paid.notnull() and amount_paid != 0")
            .rename(columns={'entry_date': 'payment_date'})
            .assign(amount_paid=lambda d: d['amount_paid'].fillna(0))
        )
        df_p.set_index(['school_year', 'student_id_pseudonimized', 'category_name'], inplace=True)

        tasks = []
        for keys, payments in df_p.groupby(level=[0, 1, 2]):
            receivables = df_cs[
                (df_cs['school_year'] == keys[0]) &
                (df_cs['student_id_pseudonimized'] == keys[1]) &
                (df_cs['category_name'] == keys[2])
            ].sort_values(by='due_date').copy()
            receivables['cs_index'] = receivables.index
            tasks.append((receivables, payments))

        with ThreadPool(processes=cpu_count()) as pool:
            results = pool.map(_allocate_payment_amounts_sequential, tasks)

        df_payment_amounts = pd.concat(results, ignore_index=True)
        df_cs = df_cs.merge(
            df_payment_amounts[['cs_index', 'prepayments', 'total_payments']],
            left_index=True,
            right_on='cs_index',
            how='left'
        ).drop(columns=['cs_index'])
        df_cs[['prepayments', 'total_payments']] = df_cs[['prepayments', 'total_payments']].fillna(0)

        return df_cs


# ─────────────────────────────────────────────────────────────────────────────
# FeatureEngineer
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Adds machine-learning features to a clean invoice DataFrame.

    Receives the output of ``InvoiceBuilder.build()`` and enriches it with
    payment-history features (DTP lags, trends, streaks), financial ratios,
    temporal signals, plan-type encodings, survival columns, and optional
    EDA-derived engineered features. Produces one row per receivable with
    all feature columns appended.

    Parameters
    ----------
    df_cs : pd.DataFrame
        Invoice-level DataFrame from ``InvoiceBuilder.build()``.
    df_revenues : pd.DataFrame
        Raw revenue records, used for cumulative balance calculations.
    df_payments_to_all : pd.DataFrame
        All payment records, used for ``days_since_last_payment``.
    df_enrollees : pd.DataFrame
        Student enrollment records providing ``plan_type``.
    args : object
        Must expose ``observation_end`` (datetime) for censoring logic.
    add_engineered_features : bool, default True
        Master switch for all EDA-derived features. Set False for exact
        backwards compatibility with the original feature set.
    add_temporal_features : bool, default True
        Adds ``due_month`` and ``due_quarter``.
    add_payment_behaviour_features : bool, default True
        Adds ``payment_ratio``, ``opening_balance_flag``, ``early_payer_flag``.
    add_streak_features : bool, default True
        Adds ``on_time_streak`` and ``prev_bracket``.
    add_dtp_summary_features : bool, default True
        Adds ``dtp_rolling_std`` and ``dtp_max``.
    add_plan_risk_score : bool, default True
        Adds ``plan_type_risk_score`` (ordinal encoding by observed 90-day rate).
    winsorise_dtp : bool, default False
        If True, clips DTP feature columns at ``winsorise_percentile`` bounds.
    winsorise_percentile : float, default 0.01
        Lower-tail percentile for winsorisation (upper = 1 − this).
        Only used when ``winsorise_dtp=True``.
    plan_risk_map : dict or None, default None
        Pre-fitted plan-type → risk-score mapping (Fit-Transform pattern).
        **Inference mode:** provide the map saved at training time so that
        new data is scored using the training-set distribution, regardless
        of batch size or composition.
        **Training mode (None):** the map is computed from the current data
        and stored in ``self.plan_risk_map`` for export into the
        ``InferencePipeline`` pickle.
    """

    def __init__(
        self,
        df_cs: pd.DataFrame,
        df_revenues: pd.DataFrame,
        df_payments_to_all: pd.DataFrame,
        df_enrollees: pd.DataFrame,
        args,
        add_engineered_features: bool = True,
        add_temporal_features: bool = True,
        add_payment_behaviour_features: bool = True,
        add_streak_features: bool = True,
        add_dtp_summary_features: bool = True,
        add_plan_risk_score: bool = True,
        winsorise_dtp: bool = False,
        winsorise_percentile: float = 0.01,
        plan_risk_map: dict | None = None,
    ):
        self.df_cs = df_cs
        self.df_revenues = df_revenues
        self.df_payments_to_all = df_payments_to_all
        self.df_enrollees = df_enrollees
        self.args = args
        self.add_engineered_features = add_engineered_features
        self.add_temporal_features = add_temporal_features
        self.add_payment_behaviour_features = add_payment_behaviour_features
        self.add_streak_features = add_streak_features
        self.add_dtp_summary_features = add_dtp_summary_features
        self.add_plan_risk_score = add_plan_risk_score
        self.winsorise_dtp = winsorise_dtp
        self.winsorise_percentile = winsorise_percentile
        # Fit-Transform state: if provided, used as-is (inference mode).
        # If None, computed from data and stored here (training mode).
        self.plan_risk_map = plan_risk_map

    def build(self) -> pd.DataFrame:
        """
        Apply all feature engineering steps and return the enriched DataFrame.

        Returns
        -------
        pd.DataFrame
            Invoice DataFrame with all ML feature columns appended.
        """
        df = self.df_cs.copy()

        # ── Core features (always computed) ───────────────────────────────────
        df = self._merge_dtp(df, dtp_n=4)

        df['dtp_avg'] = df[['dtp_1', 'dtp_2', 'dtp_3', 'dtp_4']].mean(axis=1)
        df['dtp_wavg'] = df[['dtp_1', 'dtp_2', 'dtp_3', 'dtp_4']].mul([0.4, 0.3, 0.2, 0.1]).sum(axis=1)

        df = self._merge_due_date_prev(df, 2)

        # Time-normalized trends: signed values are intentional.
        # A negative trend means the student is paying faster over time
        # (elapsed days are decreasing), which is a valid and informative
        # signal for tree-based classifiers. Do NOT take absolute value.
        df['dtp_2_trend'] = self._calculate_dtp_trend(df, lag=1)
        df['dtp_3_trend'] = self._calculate_dtp_trend(df, lag=2)

        df = self._merge_last_payment_date(df, self.df_payments_to_all)
        df['days_since_last_payment'] = (
            df['due_date'] - df['last_payment_date']
        ).dt.days.astype('Int64').fillna(-1)

        df = self._merge_amount_due_cum_sum(df, self.df_revenues)
        df = self._merge_amount_paid_cum_sum(df, self.df_revenues)
        df['opening_balance'] = (df['amount_due_cumsum'] - df['amount_paid_cumsum']).clip(lower=0)

        df = df.merge(
            self.df_enrollees[['school_year', 'student_id_pseudonimized', 'plan_type']],
            on=['school_year', 'student_id_pseudonimized'],
            how='left'
        )
        df = self._apply_one_hot_encoding(df)
        df = self._merge_dtp_bracket(df)
        df = self._merge_censor(df)
        df = self._calculate_dtp_for_censored_invoices(df)

        if self.winsorise_dtp:
            df = self._winsorise_dtp(df)

        # ── EDA-derived engineered features (opt-in, default on) ──────────────
        if self.add_engineered_features:
            if self.add_temporal_features:
                df = self._add_temporal_features(df)
            if self.add_payment_behaviour_features:
                df = self._add_payment_behaviour_features(df)
            if self.add_streak_features:
                df = self._add_streak_features(df)
            if self.add_dtp_summary_features:
                df = self._add_dtp_summary_features(df)
            if self.add_plan_risk_score:
                df = self._add_plan_risk_score(df)

        return df

    # ── Core feature methods ──────────────────────────────────────────────────

    def _merge_dtp(self, df_cs: pd.DataFrame, dtp_n: int) -> pd.DataFrame:
        """
        Extract the number of days between the current invoice due date and the
        Nth previous issued invoice (where N = no_of_invoices_back).
        Keeps date_fully_paid blank if unpaid, but still calculates dtp_# using shift.
        """
        df_cs = df_cs.sort_values(['student_id_pseudonimized', 'due_date'])
        df_cs['days_elapsed_until_fully_paid'] = (
            df_cs['date_fully_paid'] - df_cs['due_date']
        ).dt.days.astype('Int64')

        for n in range(1, dtp_n + 1):
            df_cs[f'dtp_{n}'] = (
                df_cs.groupby('student_id_pseudonimized')['days_elapsed_until_fully_paid']
                .shift(n)
                .astype('Int64')
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
        df['due_date'] = pd.to_datetime(df['due_date'])
        df = df.sort_values(['student_id_pseudonimized', 'due_date'])

        for k in range(1, n + 1):
            df[f'due_date_prev_{k}'] = (
                df.groupby('student_id_pseudonimized')['due_date'].shift(k)
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
        col_current = f'dtp_{lag + 1}'
        col_previous = 'dtp_1'
        days_diff = (df['due_date'] - df[f'due_date_prev_{lag}']).dt.days
        trend = (df[col_current] - df[col_previous]) / days_diff.replace(0, pd.NA)
        return trend.fillna(0).infer_objects(copy=False)

    def _merge_last_payment_date(
        self, df_cs: pd.DataFrame, df_p: pd.DataFrame
    ) -> pd.DataFrame:
        """
        For each receivable, get the latest payment date before its due_date,
        restricted to the same student_id_pseudonimized, but only from
        payments in a different category_name.
        """
        df_p = df_p.sort_values('payment_date').copy()
        df_cs = df_cs.sort_values('due_date').copy()

        merged = df_cs[['student_id_pseudonimized', 'category_name', 'due_date']].merge(
            df_p[['student_id_pseudonimized', 'category_name', 'payment_date']],
            on='student_id_pseudonimized',
            how='inner'
        )
        merged = merged.loc[merged['category_name_x'] != merged['category_name_y']]
        merged = merged.loc[merged['payment_date'] < merged['due_date']]

        result = (
            merged.groupby(['student_id_pseudonimized', 'due_date'])['payment_date']
            .max()
            .reset_index()
            .rename(columns={'payment_date': 'last_payment_date'})
        )
        return df_cs.merge(result, on=['student_id_pseudonimized', 'due_date'], how='left')

    def _merge_amount_due_cum_sum(
        self, df_cs: pd.DataFrame, df_revenues: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate the cumulative sum of df_revenues['amount_due'] for each student in df_cs
        where df_revenues['due_date'] <= df_cs['due_date'].
        """
        df_revenues = df_revenues.sort_values(['student_id_pseudonimized', 'due_date'])
        df_revenues['amount_due_cumsum'] = df_revenues.groupby('student_id_pseudonimized')['amount_due'].cumsum()

        merged = pd.merge(df_cs, df_revenues, on='student_id_pseudonimized', suffixes=('_cs', '_rev'))
        merged = merged[merged['due_date_rev'] <= merged['due_date_cs']]

        result = (
            merged.groupby(['student_id_pseudonimized', 'due_date_cs'], as_index=False)['amount_due_cumsum'].max()
        )
        df_cs = pd.merge(
            df_cs, result,
            left_on=['student_id_pseudonimized', 'due_date'],
            right_on=['student_id_pseudonimized', 'due_date_cs'],
            how='left'
        ).drop(columns=['due_date_cs'])
        return df_cs

    def _merge_amount_paid_cum_sum(
        self, df_cs: pd.DataFrame, df_revenues: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate the cumulative sum of df_revenues['amount_paid'] for each student in df_cs
        where df_revenues['due_date'] <= df_cs['due_date'].
        """
        df_revenues = df_revenues.sort_values(['student_id_pseudonimized', 'entry_date'])
        df_revenues['amount_paid_cumsum'] = df_revenues.groupby('student_id_pseudonimized')['amount_paid'].cumsum()

        merged = pd.merge(df_cs, df_revenues, on='student_id_pseudonimized', suffixes=('_cs', '_rev'))
        merged = merged[merged['entry_date'] <= merged['due_date_cs']]

        result = (
            merged.groupby(['student_id_pseudonimized', 'due_date_cs'], as_index=False)['amount_paid_cumsum'].max()
        )
        df_cs = pd.merge(
            df_cs, result,
            left_on=['student_id_pseudonimized', 'due_date'],
            right_on=['student_id_pseudonimized', 'due_date_cs'],
            how='left'
        ).drop(columns=['due_date_cs'])
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
        df_cs = pd.concat([df_cs, df_cs_encoded], axis=1)
        df_cs.drop(columns=categorical_features, inplace=True)
        return df_cs

    def _merge_dtp_bracket(self, df_cs: pd.DataFrame) -> pd.DataFrame:
        bracket_rules = {
            'on_time': lambda x: x <= 0,
            '30_days': lambda x: 0 < x <= 30,
            '60_days': lambda x: 30 < x <= 60,
            '90_days': lambda x: x > 60,
        }

        def assign_bracket(x):
            for label, rule in bracket_rules.items():
                if rule(x):
                    return label
            return None

        df_cs['dtp_bracket'] = df_cs['days_elapsed_until_fully_paid'].apply(assign_bracket)
        return df_cs

    def _merge_censor(self, df_cs: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a new column that is used in survival analysis.
        censor (E) = 0: The payment still has not happened yet during the observation end date
        censor (E) = 1: The exact date of full payment happened during the observation end date
        """
        observation_end_date = df_cs['due_date'].max()
        df_cs['censor'] = (df_cs['date_fully_paid'] <= observation_end_date).astype(int)
        return df_cs

    def _calculate_dtp_for_censored_invoices(self, df_cs: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate days_elapsed_until_fully_paid for censored data (censor = 0).
        Result is stored as integer days for all rows.
        """
        default_observation_end_date = datetime.today()
        observation_end_date = getattr(self.args, 'observation_end', default_observation_end_date)

        df_cs = df_cs[
            (df_cs['due_date'] <= observation_end_date) |
            ((df_cs['due_date'] > observation_end_date) & df_cs['date_fully_paid'].notna())
        ].copy()

        df_cs.loc[df_cs['censor'] == 0, 'days_elapsed_until_fully_paid'] = (
            (observation_end_date - df_cs.loc[df_cs['censor'] == 0, 'due_date']).dt.days
        )
        df_cs['days_elapsed_until_fully_paid'] = (
            df_cs['days_elapsed_until_fully_paid']
            .infer_objects(copy=False)
            .fillna(0)
            .astype(int)
        )
        return df_cs

    def _winsorise_dtp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clip DTP feature columns at the specified percentile bounds.

        Applies symmetric winsorisation: values below the lower-tail
        percentile are raised to that percentile value, and values above
        the upper-tail percentile are capped at that value. This removes
        extreme outliers (e.g. 247 DTP values above 365 days) that
        destabilise model training without discarding rows.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame with dtp_1..4, dtp_avg, dtp_wavg columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with DTP columns clipped in place.
        """
        dtp_cols = ['dtp_1', 'dtp_2', 'dtp_3', 'dtp_4', 'dtp_avg', 'dtp_wavg']
        p = self.winsorise_percentile
        for col in dtp_cols:
            if col not in df.columns:
                continue
            # dtp_1..4 are stored as nullable Int64 (capital I) because they come
            # from .astype('Int64') in _merge_dtp. quantile() returns a float, and
            # clip() cannot write float bounds back into a nullable integer column —
            # it raises TypeError. Cast to float64 first, clip, then restore Int64
            # for the lag columns so downstream code that expects integer DTP values
            # continues to work correctly.
            original_dtype = df[col].dtype
            df[col] = df[col].astype('float64')
            lo = df[col].quantile(p)
            hi = df[col].quantile(1 - p)
            df[col] = df[col].clip(lower=lo, upper=hi)
            if original_dtype == pd.Int64Dtype():
                df[col] = df[col].round().astype('Int64')
        return df

    # ── EDA-derived engineered feature methods ────────────────────────────────

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar-based features derived from due_date.

        ``due_month`` and ``due_quarter`` capture strong seasonal patterns
        identified in the EDA: June and July have 53 % and 40 % 90-day rates
        respectively, compared to the 19 % overall average, while September
        has the highest on-time rate at 54 %.

        Added columns
        -------------
        due_month : int
            Calendar month of the invoice due date (1–12).
        due_quarter : int
            Calendar quarter of the invoice due date (1–4).
        """
        df['due_month'] = df['due_date'].dt.month
        df['due_quarter'] = df['due_date'].dt.quarter
        return df

    def _add_payment_behaviour_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features that characterise a student's cumulative payment discipline.

        Added columns
        -------------
        opening_balance_flag : int (0 or 1)
            Binary indicator: 1 if opening_balance > 0. EDA showed that 83 % of
            invoices with zero opening balance are on_time — this is the single
            strongest binary separator in the dataset.
        payment_ratio : float
            amount_paid_cumsum / amount_due_cumsum. Measures what fraction of
            total charges have been settled as of the invoice due date. Values
            close to 1.0 indicate a reliable payer; values below 0.8 correlate
            strongly with the 90_days bracket.
        early_payer_flag : float (0.0 or 1.0)
            Binary indicator: 1 if dtp_1 < 0 (previous invoice was paid before
            its due date). Early payers are almost never in the 90_days bracket.
        """
        df['opening_balance_flag'] = (df['opening_balance'] > 0).astype(int)
        df['payment_ratio'] = (
            df['amount_paid_cumsum'] / df['amount_due_cumsum'].replace(0, np.nan)
        )
        df['early_payer_flag'] = (df['dtp_1'] < 0).astype(float)
        return df

    def _add_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sequence-based features that capture payment momentum per student.

        Both features require ``dtp_bracket`` to already exist in the DataFrame,
        so they must be called after ``_merge_dtp_bracket``.

        Added columns
        -------------
        on_time_streak : int
            Number of consecutive on_time invoices immediately preceding the
            current one for the same student. Resets to 0 on any non-on_time
            payment. Captures the momentum signal: students with a long streak
            are very likely to continue paying on time.
        prev_bracket : float (0–3, or NaN for first invoice)
            Integer encoding of the previous invoice's dtp_bracket for the same
            student (on_time=0, 30_days=1, 60_days=2, 90_days=3). The bracket
            transition matrix from the EDA showed 60 % on_time→on_time and
            52 % 90_days→90_days persistence, making this the highest-signal
            untested feature.
        """
        bracket_enc = {'on_time': 0, '30_days': 1, '60_days': 2, '90_days': 3}

        df = df.sort_values(['student_id_pseudonimized', 'due_date'])

        def _streak(series):
            result, count = [], 0
            for val in series:
                count = count + 1 if val == 'on_time' else 0
                result.append(count)
            # Shift by 1: streak is the count *before* current invoice
            return pd.Series(result, index=series.index).shift(1).fillna(0).astype(int)

        df['on_time_streak'] = df.groupby('student_id_pseudonimized')['dtp_bracket'].transform(_streak)

        df['prev_bracket'] = (
            df.groupby('student_id_pseudonimized')['dtp_bracket']
            .shift(1)
            .map(bracket_enc)
        )

        return df

    def _add_dtp_summary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cross-lag DTP summary statistics that complement the existing mean.

        The current feature set has dtp_avg (mean) and dtp_wavg (weighted mean),
        but lacks measures of volatility and worst-case behaviour.

        Added columns
        -------------
        dtp_rolling_std : float
            Standard deviation of dtp_1 through dtp_4. A high value indicates
            an erratic payer whose behaviour is hard to predict from the mean
            alone — this pattern is distinct from a consistently late payer.
        dtp_max : float
            Maximum value across dtp_1 through dtp_4. Captures the worst recent
            payment regardless of which invoice it came from. Complements
            dtp_avg, which masks a single very late payment.
        """
        dtp_cols = ['dtp_1', 'dtp_2', 'dtp_3', 'dtp_4']
        df['dtp_rolling_std'] = df[dtp_cols].std(axis=1)
        df['dtp_max'] = df[dtp_cols].max(axis=1)
        return df

    def _add_plan_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add an ordinal risk score derived from each plan type's observed 90-day rate.

        Fit-Transform pattern
        ---------------------
        Training mode (``self.plan_risk_map is None``):
            Computes the plan-to-score mapping from ``df`` and stores it in
            ``self.plan_risk_map`` so it can be exported and bundled with the
            deployed InferencePipeline.

        Inference mode (``self.plan_risk_map`` is a pre-fitted dict):
            Skips computation entirely and maps every row using the stored
            training-set distribution.  This ensures that a single-row
            inference batch receives the same score a plan type would have
            received at training time, regardless of the batch composition.

        Scores are computed on the labelled (censored == 1) subset only,
        so uncensored invoices without a bracket do not distort the ranking.
        Uncensored invoices are scored by lookup after the ranking is built.

        Added columns
        -------------
        plan_type_risk_score : int
            Data-driven ordinal risk score. Higher = more likely to be 90_days late.
            Rows whose plan type was not observed in the labelled set receive the
            median score (``n_quantiles // 2``).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain one-hot plan_type columns and dtp_bracket.
        n_quantiles : int
            Number of risk tiers. Default 5 gives scores 0–4, matching the
            original EDA-derived scale. Set to a smaller value (e.g. 3) if
            some plan types have too few observations for reliable ranking.
        """
        n_quantiles = 5
        fallback_score = n_quantiles // 2

        plan_ohe_map = {
            'plan_type_Plan - A': 'Plan - A',
            'plan_type_Plan - B': 'Plan - B',
            'plan_type_Plan - C': 'Plan - C',
            'plan_type_Plan - D': 'Plan - D',
            'plan_type_Plan - E': 'Plan - E',
            'plan_type_nan':      'nan',
        }
        present = [c for c in plan_ohe_map if c in df.columns]

        def _plan_label(row):
            for col in present:
                if row.get(col, 0) == 1:
                    return plan_ohe_map[col]
            return 'nan'

        df['_plan_label'] = df[present].apply(_plan_label, axis=1)

        if self.plan_risk_map is not None:
            # ── Inference / Transform mode ────────────────────────────────────
            # Use the exact distribution from training; ignore current batch.
            score_map = self.plan_risk_map
        else:
            # ── Training / Fit mode ───────────────────────────────────────────
            # Compute the map from the current data and store it for export.
            labelled = df[df['dtp_bracket'].notna()].copy()
            labelled['_is_90'] = (labelled['dtp_bracket'] == '90_days').astype(int)

            rate_per_plan = (
                labelled.groupby('_plan_label')['_is_90']
                .mean()
                .rename('_90day_rate')
            )

            if len(rate_per_plan) >= n_quantiles:
                plan_scores = pd.qcut(
                    rate_per_plan,
                    q=n_quantiles,
                    labels=False,
                    duplicates='drop',
                ).rename('plan_type_risk_score')
            else:
                plan_scores = rate_per_plan.rank(method='dense').sub(1).astype(int).rename('plan_type_risk_score')

            score_map = plan_scores.to_dict()
            # Store so CreditSalesProcessor can expose it for bundling.
            self.plan_risk_map = score_map

        df['plan_type_risk_score'] = (
            df['_plan_label']
            .map(score_map)
            .fillna(fallback_score)
            .astype(int)
        )
        df = df.drop(columns=['_plan_label'])

        return df


# ─────────────────────────────────────────────────────────────────────────────
# InvoicePostProcessor
# ─────────────────────────────────────────────────────────────────────────────

class InvoicePostProcessor:
    """
    Applies row filters, column drops, and optional descriptions to a
    fully-featured invoice DataFrame.

    Receives the output of ``FeatureEngineer.build()`` and applies all
    user-controlled cleanup: filtering out unwanted rows (back-account
    transactions, fully-paid invoices, excluded school years, records
    with missing DTP), dropping column groups, and attaching human-readable
    category descriptions. Has no knowledge of feature construction logic.

    Parameters
    ----------
    df_cs : pd.DataFrame
        Fully-featured invoice DataFrame from ``FeatureEngineer.build()``.
    drop_helper_columns : bool, default False
        Drops intermediate helper columns created during processing.
    drop_demographic_columns : bool, default False
        Drops school_year, student_id_pseudonimized, category_name.
    drop_survival_columns : bool, default False
        Drops censor and days_elapsed_until_fully_paid.
    drop_plan_type_columns : bool, default False
        Drops plan_type one-hot encoded columns.
    drop_missing_dtp : bool, default False
        Drops rows where any of dtp_1..4 is NaN.
    drop_back_account_transactions : bool, default False
        Drops rows where category_name is "Back Account".
    drop_fully_paid_invoices : bool, default False
        Drops rows where date_fully_paid is not NaT.
    add_description : bool, default False
        Adds a human-readable 'description' column from category_name.
    exclude_school_years : list or None, default None
        List of school_year values to exclude. EDA showed that 2016–2018 have
        very different label distributions (71–100 % 90_days) that may hurt
        generalisation for more recent data.
    """

    def __init__(
        self,
        df_cs: pd.DataFrame,
        drop_helper_columns: bool = False,
        drop_demographic_columns: bool = False,
        drop_survival_columns: bool = False,
        drop_plan_type_columns: bool = False,
        drop_missing_dtp: bool = False,
        drop_back_account_transactions: bool = False,
        drop_fully_paid_invoices: bool = False,
        add_description: bool = False,
        exclude_school_years: list | None = None,
    ):
        self.df_cs = df_cs
        self.drop_helper_columns = drop_helper_columns
        self.drop_demographic_columns = drop_demographic_columns
        self.drop_survival_columns = drop_survival_columns
        self.drop_plan_type_columns = drop_plan_type_columns
        self.drop_missing_dtp = drop_missing_dtp
        self.drop_back_account_transactions = drop_back_account_transactions
        self.drop_fully_paid_invoices = drop_fully_paid_invoices
        self.add_description = add_description
        self.exclude_school_years = exclude_school_years or []

    def build(self) -> pd.DataFrame:
        """
        Apply all post-processing steps and return the final DataFrame.

        Returns
        -------
        pd.DataFrame
            Cleaned, filtered invoice DataFrame ready for modelling.
        """
        df = self.df_cs.copy()
        df = self._filter_rows(df)
        df = self._drop_columns(df)
        if self.add_description:
            df = self._apply_description_function(df)
        return df

    # ── Row filters ───────────────────────────────────────────────────────────

    def _filter_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all row-level filters in a single pass."""
        if self.exclude_school_years:
            before = len(df)
            df = df.loc[~df['school_year'].isin(self.exclude_school_years)].copy()
            print(f"Excluded school years {self.exclude_school_years}: "
                  f"removed {before - len(df)} rows, {len(df)} remaining.")

        if self.drop_back_account_transactions:
            df = df.loc[df['category_name'] != 'Back Account'].copy()

        if self.drop_fully_paid_invoices:
            before = len(df)
            df = df.loc[df['date_fully_paid'].isna()].copy()
            print(f"Dropped {before - len(df)} fully paid invoices. Remaining: {len(df)}")

        if self.drop_missing_dtp:
            before = len(df)
            df = df.dropna(subset=['dtp_1', 'dtp_2', 'dtp_3', 'dtp_4'])
            print(f"Dropped {before - len(df)} invoices with missing DTP values. Remaining: {len(df)}")

        return df

    # ── Column drops ──────────────────────────────────────────────────────────

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collect all columns to drop and remove them in a single operation."""
        helper_columns = [
            'gross_receivables', 'amount_discounted', 'adjustments',
            'due_date_prev_1', 'due_date_prev_2', 'date_fully_paid', 'last_payment_date',
            # New EDA helper columns — intermediate, not ML features
            'on_time_streak', 'prev_bracket',
        ]
        demographic_columns = ['school_year', 'student_id_pseudonimized', 'category_name']
        survival_columns = ['censor', 'days_elapsed_until_fully_paid']
        plan_type_columns = [
            'plan_type_Plan - A', 'plan_type_Plan - B', 'plan_type_Plan - C',
            'plan_type_Plan - D', 'plan_type_Plan - E', 'plan_type_nan',
        ]

        cols_to_drop = []
        if self.drop_helper_columns:
            cols_to_drop.extend(helper_columns)
        if self.drop_demographic_columns:
            cols_to_drop.extend(demographic_columns)
        if self.drop_survival_columns:
            cols_to_drop.extend(survival_columns)
        if self.drop_plan_type_columns:
            cols_to_drop.extend(plan_type_columns)

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop, errors='ignore')

        return df

    # ── Description ───────────────────────────────────────────────────────────

    def _apply_description_function(self, df_cs: pd.DataFrame) -> pd.DataFrame:
        df_cs['description'] = df_cs.apply(self._get_description, axis=1)
        return df_cs

    def _get_description(self, row) -> str:
        category = row['category_name']
        if '-UE' in category:
            return 'Tuition fee (' + category[:3] + ') - Upon enrollment'
        elif 'B-1st' in category:
            return 'Tuition fee (' + category[:3] + ') - 1 of 2 payments'
        elif 'B-2nd' in category:
            return 'Tuition fee (' + category[:3] + ') - 2 of 2 payments'
        elif 'C-1st' in category:
            return 'Tuition fee (' + category[:3] + ') - 1 of 4 payments'
        elif 'C-2nd' in category:
            return 'Tuition fee (' + category[:3] + ') - 2 of 4 payments'
        elif 'C-3rd' in category:
            return 'Tuition fee (' + category[:3] + ') - 3 of 4 payments'
        elif 'C-4th' in category:
            return 'Tuition fee (' + category[:3] + ') - 4 of 4 payments'
        elif 'D-1st' in category:
            return 'Tuition fee (' + category[:3] + ') - 1 of 10 payments'
        elif 'D-2nd' in category:
            return 'Tuition fee (' + category[:3] + ') - 2 of 10 payments'
        elif 'D-3rd' in category:
            return 'Tuition fee (' + category[:3] + ') - 3 of 10 payments'
        elif 'D-4th' in category:
            return 'Tuition fee (' + category[:3] + ') - 4 of 10 payments'
        elif 'D-5th' in category:
            return 'Tuition fee (' + category[:3] + ') - 5 of 10 payments'
        elif 'D-6th' in category:
            return 'Tuition fee (' + category[:3] + ') - 6 of 10 payments'
        elif 'D-7th' in category:
            return 'Tuition fee (' + category[:3] + ') - 7 of 10 payments'
        elif 'D-8th' in category:
            return 'Tuition fee (' + category[:3] + ') - 8 of 10 payments'
        elif 'D-9th' in category:
            return 'Tuition fee (' + category[:3] + ') - 9 of 10 payments'
        elif 'D-10th' in category:
            return 'Tuition fee (' + category[:3] + ') - 10 of 10 payments'
        elif 'E-1st' in category:
            return 'Tuition fee (' + category[:3] + ') - 1 of 9 payments'
        elif 'E-2nd' in category:
            return 'Tuition fee (' + category[:3] + ') - 2 of 9 payments'
        elif 'E-3rd' in category:
            return 'Tuition fee (' + category[:3] + ') - 3 of 9 payments'
        elif 'E-4th' in category:
            return 'Tuition fee (' + category[:3] + ') - 4 of 9 payments'
        elif 'E-5th' in category:
            return 'Tuition fee (' + category[:3] + ') - 5 of 9 payments'
        elif 'E-6th' in category:
            return 'Tuition fee (' + category[:3] + ') - 6 of 9 payments'
        elif 'E-7th' in category:
            return 'Tuition fee (' + category[:3] + ') - 7 of 9 payments'
        elif 'E-8th' in category:
            return 'Tuition fee (' + category[:3] + ') - 8 of 9 payments'
        elif 'E-9th' in category:
            return 'Tuition fee (' + category[:3] + ') - 9 of 9 payments'
        elif 'E-Learning' in category:
            return 'E-learning platform (' + category[:3] + ')'
        elif '-OF-1st' in category:
            return 'Miscellaneous fees - 1 of 3 payments'
        elif '-OF-2nd' in category:
            return 'Miscellaneous fees - 2 of 3 payments'
        elif '-OF-3rd' in category:
            return 'Miscellaneous fees - 3 of 3 payments'
        elif '-OF' in category:
            return 'Miscellaneous fees'
        elif 'Books' in category:
            return 'Books (' + category[:3] + ')'
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


# ─────────────────────────────────────────────────────────────────────────────
# CreditSalesProcessor  (public-facing orchestrator — replaces CreditSales)
# ─────────────────────────────────────────────────────────────────────────────

class CreditSalesProcessor:
    """
    Orchestrates the full credit-sales processing pipeline.

    Delegates invoice construction to ``InvoiceBuilder``, feature engineering
    to ``FeatureEngineer``, and post-processing to ``InvoicePostProcessor``.
    The public interface is backwards-compatible with the original ``CreditSales``
    class — all existing call sites continue to work unchanged.

    Parameters
    ----------
    df_revenues : pd.DataFrame
        Raw revenue records. Must include an 'entry_number' column (auto-dropped).
    df_enrollees : pd.DataFrame
        Student enrollment records providing plan_type.
    args : object
        Must expose ``observation_end`` (datetime) for censoring logic.

    Invoice construction
    --------------------
    calculate_payment_amounts : bool, default False
        Adds prepayments, total_payments, adjusted_credit_amount, net_receivables.

    Post-processing filters
    -----------------------
    drop_helper_columns : bool, default False
    drop_demographic_columns : bool, default False
    drop_survival_columns : bool, default False
    drop_plan_type_columns : bool, default False
    drop_missing_dtp : bool, default False
    drop_back_account_transactions : bool, default False
    drop_fully_paid_invoices : bool, default False
    add_description : bool, default False
    exclude_school_years : list or None, default None
        School year values to exclude before modelling. Based on EDA findings,
        [2016, 2017, 2018] is a reasonable starting point — those years are
        71–100 % 90_days and have very different distributions from 2021–2025.

    Feature engineering
    -------------------
    add_engineered_features : bool, default True
        Master switch for all EDA-derived features. Set False for exact
        backwards compatibility with the original CreditSales feature set.
    add_temporal_features : bool, default True
        due_month, due_quarter.
    add_payment_behaviour_features : bool, default True
        payment_ratio, opening_balance_flag, early_payer_flag.
    add_streak_features : bool, default True
        on_time_streak, prev_bracket.
    add_dtp_summary_features : bool, default True
        dtp_rolling_std, dtp_max.
    add_plan_risk_score : bool, default True
        plan_type_risk_score.
    winsorise_dtp : bool, default False
        Clip DTP columns at percentile bounds to remove extreme outliers.
    winsorise_percentile : float, default 0.01
        Lower-tail percentile for winsorisation (upper = 1 − this).

    Fit-Transform state
    -------------------
    plan_risk_map : dict or None, default None
        Pre-fitted plan-type → risk-score mapping.
        Pass the map exported from training so that inference batches of
        any size receive the same risk scores as the training set.
        Leave as ``None`` during training — the map is then computed from
        the current data and made available via the ``plan_risk_map``
        property for bundling into the ``InferencePipeline`` pickle.
    """

    def __init__(
        self,
        df_revenues: pd.DataFrame,
        df_enrollees: pd.DataFrame,
        args,
        # ── invoice construction ──────────────────────────────────────────────
        calculate_payment_amounts: bool = False,
        # ── post-processing ───────────────────────────────────────────────────
        drop_helper_columns: bool = False,
        drop_demographic_columns: bool = False,
        drop_survival_columns: bool = False,
        drop_plan_type_columns: bool = False,
        drop_missing_dtp: bool = False,
        drop_back_account_transactions: bool = False,
        drop_fully_paid_invoices: bool = False,
        add_description: bool = False,
        exclude_school_years: list | None = None,
        # ── feature engineering ───────────────────────────────────────────────
        add_engineered_features: bool = True,
        add_temporal_features: bool = True,
        add_payment_behaviour_features: bool = True,
        add_streak_features: bool = True,
        add_dtp_summary_features: bool = True,
        add_plan_risk_score: bool = True,
        winsorise_dtp: bool = False,
        winsorise_percentile: float = 0.01,
        # ── fit-transform state ───────────────────────────────────────────────
        plan_risk_map: dict | None = None,
    ):
        df_revenues = df_revenues.drop(columns=['entry_number'])

        # ── Stage 1: build raw invoices ───────────────────────────────────────
        builder = InvoiceBuilder(
            df_revenues=df_revenues,
            calculate_payment_amounts=calculate_payment_amounts,
        )
        df_cs = builder.build()

        # ── Stage 2: engineer features ────────────────────────────────────────
        # Keep a reference so plan_risk_map can be retrieved after build().
        self._engineer = FeatureEngineer(
            df_cs=df_cs,
            df_revenues=df_revenues,
            df_payments_to_all=builder.df_payments_to_all,
            df_enrollees=df_enrollees,
            args=args,
            add_engineered_features=add_engineered_features,
            add_temporal_features=add_temporal_features,
            add_payment_behaviour_features=add_payment_behaviour_features,
            add_streak_features=add_streak_features,
            add_dtp_summary_features=add_dtp_summary_features,
            add_plan_risk_score=add_plan_risk_score,
            winsorise_dtp=winsorise_dtp,
            winsorise_percentile=winsorise_percentile,
            plan_risk_map=plan_risk_map,
        )
        df_cs = self._engineer.build()

        # ── Stage 3: post-process ─────────────────────────────────────────────
        post_processor = InvoicePostProcessor(
            df_cs=df_cs,
            drop_helper_columns=drop_helper_columns,
            drop_demographic_columns=drop_demographic_columns,
            drop_survival_columns=drop_survival_columns,
            drop_plan_type_columns=drop_plan_type_columns,
            drop_missing_dtp=drop_missing_dtp,
            drop_back_account_transactions=drop_back_account_transactions,
            drop_fully_paid_invoices=drop_fully_paid_invoices,
            add_description=add_description,
            exclude_school_years=exclude_school_years,
        )
        self.df_cs = post_processor.build()

    @property
    def plan_risk_map(self) -> dict | None:
        """
        The plan-type → risk-score mapping fitted during feature engineering.

        In training mode this is populated automatically after ``__init__``
        completes.  Export this value and bundle it into ``feature_metadata``
        of the ``InferencePipeline`` so that inference batches are scored
        using the training-set distribution rather than being recomputed.

        Returns ``None`` when ``add_plan_risk_score=False``.
        """
        return self._engineer.plan_risk_map

    def show_data(self) -> pd.DataFrame:
        """Return the fully processed DataFrame."""
        return self.df_cs
