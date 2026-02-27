import pandas as pd
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

from .data_partitioning import data_partitioning_by_due_date


class DataPreparer:
    def __init__(self, df_data, target_feature, test_size=0.2, verbose=True):
        """
        Initialize DataPreparer.

        Parameters
        ----------
        df_data : pd.DataFrame
            Input dataset.
        target_feature : str
            Target column name.
        test_size : float
            Test split ratio.
        verbose : bool
            If True, prints progress messages.
        """
        self.df_data = df_data
        self.target_feature = target_feature
        self.test_size = test_size
        self.verbose = verbose

        self.label_encoder = None
        self.class_mapping = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _log(self, message):
        """Print message only if verbose=True."""
        if self.verbose:
            print(message)

    def prep_data(self, balance_strategy="smote", undersample_threshold=0.5):
        """
        Prepare data by encoding labels, partitioning, and balancing.

        Parameters
        ----------
        balance_strategy : str
            Strategy for balancing:
            'smote', 'borderline_smote', 'smoteenn',
            'smotetomek', 'hybrid', or 'none'.
        undersample_threshold : float
            Threshold for hybrid undersampling (default=0.5).
        """

        # --- Encode target feature ---
        self.label_encoder = LabelEncoder()
        self.df_data[self.target_feature] = self.label_encoder.fit_transform(
            self.df_data[self.target_feature]
        )

        self.class_mapping = dict(
            zip(
                self.label_encoder.classes_,
                self.label_encoder.transform(self.label_encoder.classes_)
            )
        )

        # --- Partition the data ---
        self._log("Partitioning datasets based on due_date...")

        self.X_train, self.X_test, self.y_train, self.y_test, self.cut_off_date = (
            data_partitioning_by_due_date(
                self.df_data,
                target_feature=self.target_feature,
                test_size=self.test_size
            )
        )

        # --- Convert to float64 (critical for imblearn compatibility) ---
        self.X_train = pd.DataFrame(
            self.X_train.to_numpy(dtype="float64"),
            columns=self.X_train.columns,
            index=self.X_train.index
        )

        self.X_test = pd.DataFrame(
            self.X_test.to_numpy(dtype="float64"),
            columns=self.X_test.columns,
            index=self.X_test.index
        )

        # --- Apply balancing strategy ---
        if balance_strategy == "smote":
            sampler = SMOTE(random_state=42)

        elif balance_strategy == "borderline_smote":
            sampler = BorderlineSMOTE(random_state=42)

        elif balance_strategy == "smoteenn":
            sampler = SMOTEENN(random_state=42)

        elif balance_strategy == "smotetomek":
            sampler = SMOTETomek(random_state=42)

        elif balance_strategy == "hybrid":
            self._log("Applying hybrid undersample + oversample...")
            self.X_train, self.y_train = self._hybrid_balance(
                self.X_train,
                self.y_train,
                undersample_threshold
            )
            return self

        elif balance_strategy == "none":
            self._log("No balancing applied.")
            return self

        else:
            raise ValueError("Invalid balance_strategy.")

        self._log(f"Applying {balance_strategy}...")

        X_res, y_res = sampler.fit_resample(self.X_train, self.y_train)

        self.X_train = pd.DataFrame(X_res, columns=self.X_train.columns)
        self.y_train = pd.Series(y_res, name=self.target_feature)

        return self

    def _hybrid_balance(self, X, y, undersample_threshold=0.5):
        """
        Hybrid approach:
        1. Undersample majority classes down to a threshold multiple
           of the minority class.
        2. Oversample minority classes up to the new majority size.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Training labels.
        undersample_threshold : float
            Determines how aggressively majority classes are reduced.

        Returns
        -------
        X_resampled : pd.DataFrame
        y_resampled : pd.Series
        """

        class_counts = y.value_counts()
        min_size = class_counts.min()

        target_majority_size = int(min_size / undersample_threshold)

        self._log(f"Minority class size: {min_size}")
        self._log(f"Target majority size: {target_majority_size}")

        # --- Step 1: Undersample ---
        under_strategy = {
            cls: min(count, target_majority_size)
            for cls, count in class_counts.items()
        }

        under_sampler = RandomUnderSampler(
            sampling_strategy=under_strategy,
            random_state=42
        )

        X_under, y_under = under_sampler.fit_resample(X, y)

        # --- Step 2: Oversample ---
        over_strategy = {
            cls: target_majority_size
            for cls in y_under.unique()
        }

        over_sampler = RandomOverSampler(
            sampling_strategy=over_strategy,
            random_state=42
        )

        X_final, y_final = over_sampler.fit_resample(X_under, y_under)

        return (
            pd.DataFrame(X_final, columns=X.columns),
            pd.Series(y_final, name=self.target_feature)
        )

    def decode_labels(self, y_encoded):
        """
        Convert encoded labels back to original class names.

        Parameters
        ----------
        y_encoded : array-like
            Encoded labels.

        Returns
        -------
        array-like
            Original class labels.
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder not initialized. Run prep_data() first.")

        return self.label_encoder.inverse_transform(y_encoded)