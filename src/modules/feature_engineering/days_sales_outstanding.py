from datetime import datetime
import pandas as pd
from pandas.tseries.offsets import MonthEnd, MonthBegin

class DSO:
    def __init__(self, df_all_transactions, df_credit_sales):
        # List of unique end-of-month dates from due_date column
        dates = self._get_end_of_month_dates(df_all_transactions, 'due_date')

        # Calculate running_ar column
        records = []
        for date in dates:
            running_receivable = self._get_running_receivables(df_all_transactions, date)
            records.append({"date": date, "running_receivable": running_receivable})
        
        df_running_ar = pd.DataFrame(records, columns=["date", "running_receivable"])
        df_running_ar.set_index("date", inplace=True)

        # Calculate credit_sales column
        credit_sale_rows = []
        
        for end_date in dates:
            beginning_date = end_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            credit_sale = self._get_credit_sales(df_credit_sales, beginning_date, end_date)
            credit_sale_rows.append({"date": end_date, "credit_sale": credit_sale})
        
        df_credit_sales_monthly = pd.DataFrame(credit_sale_rows, columns=["date", "credit_sale"])
        df_credit_sales_monthly.set_index("date", inplace=True)

        # Merge running_ar column and credit sales column
        df_ar_stats = df_running_ar.join(df_credit_sales_monthly, how="inner")

        self.df_dso = self._calculate_dso(df_ar_stats)
        
        
    def _get_running_receivables(self, df_revenues: pd.DataFrame, date: datetime) -> int:
        df_ar = df_revenues[['due_date', 'receivables']]
        df_ar = df_ar[df_ar['due_date'] <= date]
        running_ar = df_ar['receivables'].sum()
    
        return running_ar
    
    def _get_credit_sales(self, df_credit_sales: pd.DataFrame, date_beginning: datetime, date_ending: datetime) -> int:
        df_cs = df_credit_sales[['due_date', 'adjusted_credit_amount']]
        df_cs = df_cs[(df_cs['due_date'] >= date_beginning) & (df_cs['due_date'] <= date_ending)]
    
        credit_sale_amount = df_cs['adjusted_credit_amount'].sum()
    
        return credit_sale_amount
    
    def _get_end_of_month_dates(self, df_all_tranaactions: pd.DataFrame, date_column: str) -> list:
        """
        Return a single-column DataFrame of unique end-of-month dates
        derived from the specified date column.
    
        Parameters
        ----------
        df_all_tranaactions : pd.DataFrame
            Input DataFrame containing a date column.
        date_column : str
            Name of the column in df_all_tranaactions that contains dates.
    
        Returns
        -------
        pd.DataFrame
            A one-column DataFrame with unique end-of-month dates,
            sorted in ascending order.
        """
        # Ensure datetime
        dates = pd.to_datetime(df_all_tranaactions[date_column], errors="coerce")
    
        # Compute month-end
        end_of_month = dates + pd.offsets.MonthEnd(0)
    
        # Keep unique and sort
        unique_eom = sorted(end_of_month.dropna().unique())
    
        return unique_eom
        
    def _calculate_dso(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Days Sales Outstanding (DSO) contributions based on
        average receivables between consecutive periods and the date interval.
        Also computes a 12-month rolling DSO using first/last receivable
        balances in the window (since running_receivable is cumulative).
    
        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns ['date', 'running_receivable', 'credit_sale'].
            'date' should be datetime-like and sorted ascending.
    
        Returns
        -------
        pd.DataFrame
            Original DataFrame with extra columns:
            - 'dso_component': monthly DSO contribution for each interval
            - 'rolling_12m_dso': trailing 12-month DSO at each date
            - 'rolling_12m_dso_pct_change': % change vs. previous period
        """
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index().reset_index().rename(columns={"index": "date"})
    
        # Ensure sorted by date
        df = df.sort_values("date").reset_index(drop=True)
    
        # Previous receivable and date
        df["prev_receivable"] = df["running_receivable"].shift(1)
        df["prev_date"] = df["date"].shift(1)
    
        # Average receivable between periods
        df["avg_receivable"] = (df["running_receivable"] + df["prev_receivable"]) / 2
    
        # Interval in days
        df["days_interval"] = (df["date"] - df["prev_date"]).dt.days
    
        # Monthly DSO component
        df["dso_component"] = (
            df["avg_receivable"] / df["credit_sale"] * df["days_interval"]
        )
    
        # --- 12-month rolling DSO ---
        rolling_dso = []
        for i, current_date in enumerate(df["date"]):
            cutoff = (current_date - pd.DateOffset(months=11)).replace(day=1)
            mask = (df["date"] > cutoff) & (df["date"] <= current_date)
    
            if mask.sum() > 0:
                first_val = df.loc[mask, "running_receivable"].iloc[0]
                last_val = df.loc[mask, "running_receivable"].iloc[-1]
                avg_receivable = (first_val + last_val) / 2
    
                denom = df.loc[mask, "credit_sale"].sum()
                days_span = df.loc[mask, "days_interval"].sum()
    
                rolling_dso.append(avg_receivable / denom * days_span if denom != 0 else pd.NA)
            else:
                rolling_dso.append(pd.NA)
    
        df["rolling_12m_dso"] = rolling_dso
    
        # --- % change vs. previous period ---
        df["rolling_12m_dso_pct_change"] = df["rolling_12m_dso"].pct_change() * 100
    
        return df

    def show_data(self):
        return self.df_dso