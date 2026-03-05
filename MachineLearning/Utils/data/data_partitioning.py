import pandas as pd

def data_partitioning_by_due_date(df, target_feature, test_size=0.2):
    """
    Splits dataframe into train/test sets based on recent due_date values.
    
    Parameters:
        df (pd.DataFrame): Input dataset (must include 'due_date' column)
        target_feature (str): Name of target column
        test_size (float): Proportion for test split
    
    Returns:
        X_train, X_test, y_train, y_test, cutoff_date
    """
    # Ensure due_date is datetime
    df = df.copy()
    df['due_date'] = pd.to_datetime(df['due_date'])
    
    # Sort by due_date ascending
    df_sorted = df.sort_values(by='due_date')
    
    # Determine split index
    split_index = int(len(df_sorted) * (1 - test_size))
    
    # Partition
    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]
    
    # Drop target and due_date from features
    X_train = train_df.drop(columns=[target_feature, 'due_date'])
    y_train = train_df[target_feature]
    
    X_test = test_df.drop(columns=[target_feature, 'due_date'])
    y_test = test_df[target_feature]
    
    # Cutoff date is the first due_date in the test set
    cutoff_date = test_df['due_date'].min()
    
    return X_train, X_test, y_train, y_test, cutoff_date