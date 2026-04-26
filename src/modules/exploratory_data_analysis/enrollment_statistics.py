import pandas as pd

def enrollment_statistics(df: pd.DataFrame, mode: str = "percent") -> pd.DataFrame:
    """
    Computes yearly statistics of consecutive enrollment streaks.

    Parameters
    ----------
    df : pd.DataFrame
        Must include columns:
        ['school_year', 'student_id_pseudonimized', 'consecutive_years', 'has_refunded']
    mode : str, optional
        "percent" (default) → percentages of total enrolled
        "count" → raw counts

    Returns
    -------
    pd.DataFrame
        Table with school_year, total_enrolled, new_enrollees, and streak breakdowns.
    """

    # Work only with non-refunded rows
    df_no_refund = df[df['has_refunded'] != "Has Refund"].copy()
    df_no_refund['school_year'] = df_no_refund['school_year'].astype(int)

    # Total enrolled per year (unique students)
    totals = df_no_refund.groupby('school_year')['student_id_pseudonimized'].nunique().rename("total_enrolled")

    # Count how many students fall into each consecutive_year bucket
    counts = (
        df_no_refund.groupby(['school_year', 'consecutive_years'])['student_id_pseudonimized']
        .nunique()
        .unstack(fill_value=0)
    )

    # For each student, find their first year of appearance
    first_year = (
        df_no_refund.groupby('student_id_pseudonimized')['school_year']
        .min()
        .rename("first_year")
    )
    df_with_first = df_no_refund.merge(first_year, on="student_id_pseudonimized", how="left")
    new_counts = (
        df_with_first[df_with_first['school_year'] == df_with_first['first_year']]
        .groupby('school_year')['student_id_pseudonimized']
        .nunique()
        .rename("new_enrollees")
    )

    # Merge totals and new_enrollees
    result = pd.concat([totals, new_counts], axis=1).fillna(0)

    # Add streak counts
    result = result.merge(counts, left_index=True, right_index=True, how="left").fillna(0)

    # Convert to percentages if requested
    if mode == "percent":
        for col in result.columns:
            if col not in ["total_enrolled"]:
                result[col] = (result[col] / result["total_enrolled"]) * 100

    # Rename streak columns for clarity
    rename_map = {c: f"{int(c)}_year" + ("_%" if mode=="percent" else "_count")
                  for c in result.columns if isinstance(c, (int, float))}
    result = result.rename(columns=rename_map)

    # Rename new_enrollees column
    result = result.rename(columns={"new_enrollees": "new_enrollees_%"
                                    if mode=="percent" else "new_enrollees_count"})

    return result.reset_index()