import pandas as pd
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

from .data_partitioning import data_partitioning_by_due_date
from machine_learning.utils.balancing.hybrid_balance import HybridBalance
from machine_learning.utils.data.normalize_data import normalize
from machine_learning.utils.features.lda_transformer import LDATransformer


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
        self.cut_off_date  = None
        self.lda_transformer = None   # populated by apply_lda()

        # Outputs
        self.X_train = None
        self.X_test  = None   # Optional — None when training on the full dataset
        self.y_train = None
        self.y_test  = None   # Optional — None when training on the full dataset

    def _log(self, message):
        if self.verbose:
            print(message)

    def encode_labels(self):
        """
        Encode the target feature column using a fixed ordinal ordering.

        Labels are assigned integers in ascending severity order:
        on_time=0, 30_days=1, 60_days=2, 90_days=3. This ordering is
        required by OrdinalClassifier, which trains binary classifiers
        on thresholds P(y > k) — incorrect ordering would invert the
        ordinal structure and produce meaningless cumulative probabilities.

        Populates self.label_encoder and self.class_mapping.

        Returns
        -------
        self
        """
        ORDINAL_ORDER = ["on_time", "30_days", "60_days", "90_days"]

        self.df_data[self.target_feature] = pd.Categorical(
            self.df_data[self.target_feature],
            categories=ORDINAL_ORDER,
            ordered=True,
        ).codes  # assigns 0, 1, 2, 3 in defined order

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(ORDINAL_ORDER)  # keeps decode_labels() functional

        self.class_mapping = {cat: i for i, cat in enumerate(ORDINAL_ORDER)}
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
        self.y_test  = y_test_raw

        return self

    def use_full_dataset(self):
        """
        Skip train/test split and use the entire dataset for training.

        Selects only numeric columns and casts to float64 — the same
        dtype contract that train_test_split enforces — so that resample()
        and normalize() work correctly without a split.

        X_test / y_test are left as None; normalize() already guards
        against this.
        """
        self._log("Using full dataset for training (no split)...")

        X = self.df_data.drop(columns=[self.target_feature])
        y = self.df_data[self.target_feature]

        # Keep only numeric columns and cast to float64,
        # matching the dtype contract of train_test_split
        numeric_cols = X.select_dtypes(include=["number"]).columns
        X = pd.DataFrame(
            X[numeric_cols].to_numpy(dtype="float64"),
            columns=numeric_cols,
            index=X.index,
        )

        self.X_train = X
        self.y_train = y
        # X_test / y_test intentionally left as None

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
            "smote":            SMOTE(random_state=42),
            "borderline_smote": BorderlineSMOTE(random_state=42),
            "smote_enn":        SMOTEENN(random_state=42),
            "smote_tomek":      SMOTETomek(random_state=42),
            "hybrid":           HybridBalance(undersample_threshold=undersample_threshold, random_state=42),
            "none":             None,
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
        Fit StandardScaler on training data and transform both train and test
        sets.

        X_test is optional — when None (e.g. full-dataset finalization in
        Step 5), only X_train is normalized and X_test is left as None.
        """
        numeric_cols = self.X_train.select_dtypes(include=["float64", "int64"]).columns

        self.X_train[numeric_cols] = normalize(self.X_train[numeric_cols])

        if self.X_test is not None:
            self.X_test[numeric_cols] = normalize(self.X_test[numeric_cols])

        return self

    def apply_lda(self, mode: str = "append", n_components: int | None = None,
                  verbose: bool = False):
        """
        Fit an LDATransformer on the training set and project both train
        and test sets into LDA space.

        Must be called AFTER encode_labels() and train_test_split() (or
        use_full_dataset()), because LDA is a supervised transform that
        requires labelled training data.

        Call order in a typical pipeline::

            preparer.encode_labels()
                     .train_test_split()
                     .resample(...)
                     .apply_lda(mode="append")   # ← before normalize()
                     .normalize()

        Parameters
        ----------
        mode : {"append", "replace"}
            "append"  → keep all original features and add LD1 … LDk columns.
                        Best for tree-based models (XGBoost, RF) that can
                        exploit both raw and compressed signals.
            "replace" → return only the LD columns.
                        Best for linear or distance-based models (KNN, LR)
                        that benefit from a compact, decorrelated input.
        n_components : int or None
            Number of discriminant components.  None uses the maximum
            possible (n_classes - 1 = 3 for your 4-bracket target).
        verbose : bool
            If True, prints the explained separation variance per component.

        Returns
        -------
        self
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError(
                "apply_lda() requires X_train and y_train to be set. "
                "Call train_test_split() or use_full_dataset() first."
            )

        self._log(f"Applying LDA (mode='{mode}')...")

        self.lda_transformer = LDATransformer(
            n_components=n_components,
            mode=mode,
            verbose=verbose,
        )

        # Fit on training data only — never on test data
        self.X_train = self.lda_transformer.fit_transform(self.X_train, self.y_train)

        # Transform test set if it exists (eval runs); None for full-dataset runs
        if self.X_test is not None:
            self.X_test = self.lda_transformer.transform(self.X_test)

        self._log(
            f"  LDA done — X_train shape: {self.X_train.shape}, "
            f"LD columns: {self.lda_transformer.ld_columns_}"
        )

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
            .resample(balance_strategy=balance_strategy, undersample_threshold=undersample_threshold)
            .normalize()
        )

    def prep_full_data(self, balance_strategy="smote", undersample_threshold=0.5):
        """Full pipeline without split: encode → full dataset → resample → normalize."""
        return (
            self
            .encode_labels()
            .use_full_dataset()
            .resample(balance_strategy=balance_strategy, undersample_threshold=undersample_threshold)
            .normalize()
        )

    def prep_data_with_lda(
        self,
        balance_strategy: str = "smote",
        undersample_threshold: float = 0.5,
        lda_mode: str = "append",
        lda_n_components: int | None = None,
        lda_verbose: bool = False,
    ):
        """
        Full preparation pipeline with LDA:
        encode → split → resample → LDA → normalize.

        LDA is applied after resampling so that synthetic SMOTE samples
        are included in the LDA fit, and before normalization so that
        StandardScaler inside LDATransformer and the outer normalize()
        do not double-scale.

        Parameters
        ----------
        balance_strategy : str
            Resampling strategy passed to resample().
        undersample_threshold : float
            Threshold for hybrid undersampling.
        lda_mode : {"append", "replace"}
            Passed to apply_lda().
        lda_n_components : int or None
            Number of LDA components.  None → max possible (3).
        lda_verbose : bool
            Print LDA variance summary if True.

        Returns
        -------
        self
        """
        return (
            self
            .encode_labels()
            .train_test_split()
            .resample(balance_strategy=balance_strategy,
                      undersample_threshold=undersample_threshold)
            .apply_lda(mode=lda_mode, n_components=lda_n_components,
                       verbose=lda_verbose)
            .normalize()
        )

    def prep_full_data_with_lda(
        self,
        balance_strategy: str = "smote",
        undersample_threshold: float = 0.5,
        lda_mode: str = "append",
        lda_n_components: int | None = None,
        lda_verbose: bool = False,
    ):
        """
        Full pipeline without train/test split, with LDA:
        encode → full dataset → resample → LDA → normalize.

        Used by FinalizationRunner for deployment-ready models.

        Returns
        -------
        self
        """
        return (
            self
            .encode_labels()
            .use_full_dataset()
            .resample(balance_strategy=balance_strategy,
                      undersample_threshold=undersample_threshold)
            .apply_lda(mode=lda_mode, n_components=lda_n_components,
                       verbose=lda_verbose)
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