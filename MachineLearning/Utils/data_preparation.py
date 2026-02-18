import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

class DataPreparer:
    def __init__(self, df_data, target_feature, test_size=0.2):
        self.df_data = df_data
        self.target_feature = target_feature
        self.test_size = test_size
        self.label_encoder = None
        self.class_mapping = None

    def prep_data(self):
        # --- Encode target feature ---
        self.label_encoder = LabelEncoder()
        self.df_data[self.target_feature] = self.label_encoder.fit_transform(
            self.df_data[self.target_feature]
        )

        # Store mapping of encoded labels to original classes
        self.class_mapping = dict(
            zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_))
        )

        # --- Partition the data ---
        print("Partitioning the datasets...")
        X = self.df_data.drop(columns=[self.target_feature])
        y = self.df_data[self.target_feature]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )

        # --- Manual oversampling for multiclass ---
        print("Applying manual oversampling...")
        train_df = pd.concat([self.X_train, self.y_train], axis=1)

        dfs = []
        max_size = train_df[self.target_feature].value_counts().max()

        for cls in train_df[self.target_feature].unique():
            df_cls = train_df[train_df[self.target_feature] == cls]
            df_resampled = resample(
                df_cls,
                replace=True,
                n_samples=max_size,
                random_state=42
            )
            dfs.append(df_resampled)

        balanced_train = pd.concat(dfs)

        # Split back into X and y
        self.X_train = balanced_train.drop(columns=[self.target_feature])
        self.y_train = balanced_train[self.target_feature]

        return self

    def decode_labels(self, y_encoded):
        """Convert encoded labels back to original class names."""
        if self.label_encoder is None:
            raise ValueError("Label encoder not initialized. Run prep_data() first.")
        return self.label_encoder.inverse_transform(y_encoded)