import pandas as pd
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

from .data_partitioning import data_partitioning_by_due_date
from machine_learning.utils.balancing.hybrid_balance import HybridBalance
from machine_learning.utils.data.normalize_data import normalize


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
        self.cut_off_date = None

        # Outputs
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _log(self, message):
        if self.verbose:
            print(message)

    def encode_labels(self):
        """
        Encode the target feature column using LabelEncoder.

        Populates self.label_encoder and self.class_mapping.

        Returns
        -------
        self
        """
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
        return self

    def train_test_split(self):
        """
        Partition data into train/test sets based on due_date.

        Populates self.X_train, self.X_test, self.y_train,
        self.y_test, and self.cut_off_date.

        Returns
        -------
        self
        """
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

        return self

    def resample(self, balance_strategy="smote", undersample_threshold=0.5):
        """
        Balance the training set using the specified resampling strategy.

        Parameters
        ----------
        strategy : str
            One of: 'smote', 'borderline_smote', 'smote_enn',
            'smote_tomek', 'hybrid', or 'none'.
        undersample_threshold : float
            Threshold for hybrid undersampling (default=0.5).

        Returns
        -------
        self
        """
        samplers = {
            "smote": SMOTE(random_state=42),
            "borderline_smote": BorderlineSMOTE(random_state=42),
            "smote_enn": SMOTEENN(random_state=42),
            "smote_tomek": SMOTETomek(random_state=42),
            "hybrid": HybridBalance(undersample_threshold=undersample_threshold, random_state=42),
            "none": None,
        }

        if balance_strategy not in samplers:
            raise ValueError(
                f"Invalid strategy '{balance_strategy}'. "
                f"Choose from: {list(samplers.keys())}"
            )

        sampler = samplers[balance_strategy]

        if sampler is None:
            self._log("No balancing applied.")
        else:
            self._log(f"Applying {balance_strategy}...")
            X_res, y_res = sampler.fit_resample(self.X_train, self.y_train)
            self.X_train = pd.DataFrame(X_res, columns=self.X_train.columns)
            self.y_train = pd.Series(y_res, name=self.target_feature)

        return self

    def normalize(self):
        """
        Fit StandardScaler on training data and transform both
        train and test sets.

        Returns
        -------
        self
        """
        numeric_cols = self.X_train.select_dtypes(include=["float64", "int64"]).columns

        self.X_train[numeric_cols] = normalize(self.X_train[numeric_cols])
        self.X_test[numeric_cols] = normalize(self.X_test[numeric_cols])
        return self

    def prep_data(self, balance_strategy="smote", undersample_threshold=0.5):
        """
        Run the full preparation pipeline:
        encode labels → split → resample → normalize.

        Parameters
        ----------
        balance_strategy : str
            Resampling strategy passed to resample().
        undersample_threshold : float
            Threshold for hybrid undersampling (default=0.5).

        Returns
        -------
        self
        """
        return (
            self
            .encode_labels()
            .train_test_split()
            .resample(strategy=balance_strategy, undersample_threshold=undersample_threshold)
            .normalize()
        )

    def decode_labels(self, y_encoded):
        """
        Inverse-transform encoded labels back to original classes.

        Parameters
        ----------
        y_encoded : array-like
            Encoded label values.

        Returns
        -------
        np.ndarray
            Original class labels.
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder not initialized. Run encode_labels() first.")
        return self.label_encoder.inverse_transform(y_encoded)