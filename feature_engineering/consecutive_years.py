import pandas as pd

def get_consecutive_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column 'consecutive_years' that counts the number of consecutive
    school years each student has been enrolled, while keeping refunded rows
    in the final DataFrame. Refund rows do not contribute to streaks.
    """

    # Work only on non-refunded rows for streak calculation
    df_no_refund = df[df['has_refunded'] != "Has Refund"].copy()
    df_no_refund['school_year'] = df_no_refund['school_year'].astype(int)
    df_no_refund = df_no_refund.sort_values(['student_id_pseudonimized', 'school_year'])

    def compute_streaks(group):
        streaks = []
        prev_year = None
        current_streak = 0
        for year in group['school_year']:
            if prev_year is not None and year == prev_year + 1:
                current_streak += 1
            else:
                current_streak = 1
            streaks.append(current_streak)
            prev_year = year
        group['consecutive_years'] = streaks
        return group

    # Calculate streaks only on non-refunded rows
    df_no_refund = df_no_refund.groupby('student_id_pseudonimized', group_keys=False).apply(compute_streaks)

    # Merge streaks back into the original df (refund rows will get NaN)
    df = df.merge(
        df_no_refund[['student_id_pseudonimized', 'school_year', 'consecutive_years']],
        on=['student_id_pseudonimized', 'school_year'],
        how='left'
    )

    return df