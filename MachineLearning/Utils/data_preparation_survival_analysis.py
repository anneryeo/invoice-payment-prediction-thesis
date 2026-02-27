import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

from .data_partitioning import data_partitioning_by_due_date


class SurvivalDataPreparer:
    def __init__(self, df_data, target_feature,
                 time_feature="days_elapsed_until_fully_paid",
                 censor_feature="censor",
                 test_size=0.2, verbose=True):
        """
        Initialize SurvivalDataPreparer.

        Parameters
        ----------
        df_data : pd.DataFrame
            Input dataset.
        target_feature : str
            Target column name (classification labels).
        time_feature : str
            Column name for time-to-event.
        censor_feature : str
            Column name for censor indicator (1=event, 0=censored).
        test_size : float
            Test split ratio.
        verbose : bool
            If True, prints progress messages.
        """
        self.df_data = df_data.copy()
        self.target_feature = target_feature
        self.time_feature = time_feature
        self.censor_feature = censor_feature
        self.test_size = test_size
        self.verbose = verbose

        self.label_encoder = None
        self.class_mapping = None

        # Outputs
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.T_train = None
        self.T_test = None
        self.E_train = None
        self.E_test = None

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


        # --- Balance training data (X, y only) ---

        # Survival variables before balancing
        T_train = self.df_data.loc[self.X_train.index, self.time_feature]
        E_train = self.df_data.loc[self.X_train.index, self.censor_feature]
        T_test = self.df_data.loc[self.X_test.index, self.time_feature]
        E_test = self.df_data.loc[self.X_test.index, self.censor_feature]

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
            self.X_train, self.y_train, self.T_train, self.E_train = self._hybrid_balance(
                self.X_train, self.y_train, T_train, E_train, undersample_threshold
            )
        elif balance_strategy == "none":
            self._log("No balancing applied.")
        else:
            raise ValueError("Invalid balance_strategy.")

        # Apply sampler that is assigned
        if balance_strategy not in ["hybrid", "none"]:
            self._log(f"Applying {balance_strategy}...")

            # Concatenate survival variables with X_train
            Xy_train = self.X_train.copy()
            Xy_train[self.time_feature] = T_train
            Xy_train[self.censor_feature] = E_train

            # Resample including survival variables
            X_res, y_res = sampler.fit_resample(Xy_train, self.y_train)

            # Split back out
            self.X_train = X_res.drop([self.time_feature, self.censor_feature], axis=1)
            self.T_train = X_res[self.time_feature]
            self.E_train = X_res[self.censor_feature]
            self.y_train = pd.Series(y_res, name=self.target_feature)
        elif balance_strategy == "hybrid":
            self.X_train = self.X_train.drop([self.time_feature, self.censor_feature], axis=1)
        else:
            # If none, just assign directly
            # Since the helper function already assigns it
            self.T_train = T_train
            self.E_train = E_train

        # Test set survival variables remain unchanged
        self.T_test = T_test
        self.E_test = E_test


        # --- Normalize the data ---
        numeric_cols = self.X_train.select_dtypes(include=["float64", "int64"]).columns
        scaler = StandardScaler()
        self.X_train[numeric_cols] = scaler.fit_transform(self.X_train[numeric_cols])
        self.X_test[numeric_cols] = scaler.transform(self.X_test[numeric_cols])

        # Shift payment periods to have a minimum of 0
        ε = 1e-6
        self.T_train = self._adjust_payment_period(self.T_train, ε)
        self.T_test = self._adjust_payment_period(self.T_test, ε)

        # Ensure X_test also drops survival variables
        self.X_test = self.X_test.drop(
            [self.time_feature, self.censor_feature],
            axis=1,
            errors="ignore"
        )

        return self

    def _hybrid_balance(self, X, y, T, E, undersample_threshold=0.5):
        """Hybrid undersample + oversample strategy with survival variables."""

        # Concatenate survival variables into X
        Xy = X.copy()
        Xy[self.time_feature] = T
        Xy[self.censor_feature] = E

        # Compute class sizes
        class_counts = y.value_counts()
        min_size = class_counts.min()
        target_majority_size = int(min_size / undersample_threshold)

        self._log(f"Minority class size: {min_size}")
        self._log(f"Target majority size: {target_majority_size}")

        # Undersampling strategy
        under_strategy = {
            cls: min(count, target_majority_size)
            for cls, count in class_counts.items()
        }
        under_sampler = RandomUnderSampler(
            sampling_strategy=under_strategy, random_state=42
        )
        X_under, y_under = under_sampler.fit_resample(Xy, y)

        # Oversampling strategy
        over_strategy = {cls: target_majority_size for cls in y_under.unique()}
        over_sampler = RandomOverSampler(
            sampling_strategy=over_strategy, random_state=42
        )
        X_final, y_final = over_sampler.fit_resample(X_under, y_under)

        # Split back out so T and E match X
        X_balanced = X_final.drop([self.time_feature, self.censor_feature], axis=1)
        T_balanced = X_final[self.time_feature].reset_index(drop=True)
        E_balanced = X_final[self.censor_feature].reset_index(drop=True)

        return (
            pd.DataFrame(X_balanced, columns=X.columns),
            pd.Series(y_final, name=self.target_feature),
            T_balanced,
            E_balanced
        )
    
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
        # Maximum to only get pre-payments (negative or zero values)
        earliest_pre_payment = np.minimum(T, 0)
        
        # Shift T to avoid negatives and add epsilon to avoid zero
        adjusted_T = T - earliest_pre_payment + ε
        
        return adjusted_T

    def decode_labels(self, y_encoded):
        if self.label_encoder is None:
            raise ValueError("Label encoder not initialized. Run prep_data() first.")
        return self.label_encoder.inverse_transform(y_encoded)