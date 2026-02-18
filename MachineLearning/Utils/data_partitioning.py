from sklearn.model_selection import train_test_split

def data_partitioning(df, target_feature, test_size=0.2, random_state=42):
    """
    Splits dataframe into train/test sets.
    
    Parameters:
        df (pd.DataFrame): Input dataset
        target_col (str): Name of target column
        test_size (float): Proportion for test split
        random_state (int): Random seed
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_feature])
    y = df[target_feature]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test