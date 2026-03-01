import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

from .data_partitioning import data_partitioning_by_due_date
from .hybrid_balance import HybridBalance


class DataPreparer:
    def __init__(self, df_data, target_feature,
                 test_size=0.2, verbose=True):
        """
        Initialize DataPreparer.

        Parameters
        ----------
        df_data : pd.DataFrame
            Input dataset.
        target_feature : str
            Target column name (classification labels).
        test_size : float
            Test split ratio.
        verbose : bool
            If True, prints progress messages.
        """
        self.df_data = df_data.copy()
        self.target_feature = target_feature
        self.test_size = test_size
        self.verbose = verbose

        self.label_encoder = None
        self.class_mapping = None

        # Outputs
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _log(self, message):
        if self.verbose:
            print(message)

    def prep_data(self, balance_strategy="smote", undersample_threshold=0.5):
        """
        Prepare data by encoding labels, partitioning, balancing,
        and aligning survival variables.

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

        X_train_raw, X_test_raw, y_train_raw, y_test_raw, self.cut_off_date = (
            data_partitioning_by_due_date(
                self.df_data,
                target_feature=self.target_feature,
                test_size=self.test_size
            )
        )

        self.X_train = pd.DataFrame(
            X_train_raw.to_numpy(dtype="float64"),
            columns=X_train_raw.columns,
            index=X_train_raw.index
        )
        self.X_test = pd.DataFrame(
            X_test_raw.to_numpy(dtype="float64"),
            columns=X_test_raw.columns,
            index=X_test_raw.index
        )
        self.y_train = y_train_raw
        self.y_test = y_test_raw

        # --- Balance training data ---
        samplers = {
            "smote": SMOTE(random_state=42),
            "borderline_smote": BorderlineSMOTE(random_state=42),
            "smoteenn": SMOTEENN(random_state=42),
            "smotetomek": SMOTETomek(random_state=42),
            "hybrid": HybridBalance(undersample_threshold=undersample_threshold, random_state=42),
            "none": None,
        }

        if balance_strategy not in samplers:
            raise ValueError("Invalid balance_strategy.")

        sampler = samplers[balance_strategy]

        if sampler is None:
            self._log("No balancing applied.")
        else:
            self._log(f"Applying {balance_strategy}...")
            X_res, y_res = sampler.fit_resample(self.X_train, self.y_train)
            self.X_train = pd.DataFrame(X_res, columns=self.X_train.columns)
            self.y_train = pd.Series(y_res, name=self.target_feature)

        # --- Normalize the data ---
        numeric_cols = self.X_train.select_dtypes(include=["float64", "int64"]).columns
        scaler = StandardScaler()
        self.X_train[numeric_cols] = scaler.fit_transform(self.X_train[numeric_cols])
        self.X_test[numeric_cols] = scaler.transform(self.X_test[numeric_cols])

        return self

    def _adjust_payment_period(self, T, ε=1e-6):
        """
        Adjusts the payment period to avoid negative or zero values.

        Parameters
        ----------
        T : array-like or float
            The time period(s) to adjust.
        epsilon : float, optional
            A small constant added to avoid zero values (default is 1e-6).

        Returns
        -------
        adjusted_T : ndarray or float
            The adjusted time period(s).
        earliest_pre_payment : ndarray or float
            The portion representing pre-payments (non-positive values).
        """
        earliest_pre_payment = np.minimum(T, 0)
        adjusted_T = T - earliest_pre_payment + ε
        return adjusted_T

    def decode_labels(self, y_encoded):
        if self.label_encoder is None:
            raise ValueError("Label encoder not initialized. Run prep_data() first.")
        return self.label_encoder.inverse_transform(y_encoded)