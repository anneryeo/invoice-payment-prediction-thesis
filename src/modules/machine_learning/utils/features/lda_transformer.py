"""
lda_transformer.py
==================
Standalone LDA dimensionality-reduction helper.

Wraps sklearn's LinearDiscriminantAnalysis in a TransformerMixin so it
can be:
  - used standalone via LDATransformer.fit_transform(X, y)
  - slotted directly into any sklearn Pipeline as a step
  - imported into DataPreparer via DataPreparer.apply_lda()

Design decisions
----------------
* log1p is applied to right-skewed financial features BEFORE LDA so that
  LDA's Gaussian assumptions are better satisfied.  DTP features are skipped
  because they can be negative (early payers).
* Features are StandardScaled inside the transformer so the caller does not
  need to scale first.
* Components are returned as a DataFrame with columns LD1, LD2, LD3 so they
  are immediately legible in downstream code.
* mode="append" (default) keeps all original features AND adds the LD
  columns.  mode="replace" returns only the LD columns.  Append is usually
  better for tree-based models; replace is better for linear/distance-based
  models.

Usage
-----
Standalone::

    from lda_transformer import LDATransformer

    transformer = LDATransformer(mode="append")
    transformer.fit(X_train, y_train)
    X_train_out = transformer.transform(X_train)
    X_test_out  = transformer.transform(X_test)

Inside DataPreparer (added as apply_lda())::

    preparer = DataPreparer(df, target_feature="dtp_bracket")
    preparer.prep_data(balance_strategy="smote")
    preparer.apply_lda(mode="append")
    # preparer.X_train and preparer.X_test now have LD1/LD2/LD3 columns

Inside TwoStageClassifier (automatic — no caller config needed)::

    # Stage 1: fitted on all 4 classes → produces LD1/LD2/LD3 (3 components)
    # Stage 2: fitted on delinquent rows only (y > 0, remapped to 0/1/2)
    #          → produces LD1/LD2 (2 components), optimised for 30/60/90-day
    #          separation without on_time noise
    #
    # The same LDATransformer class handles both because n_classes_ is inferred
    # from y at fit() time rather than being hardcoded.

Inside an sklearn Pipeline::

    from sklearn.pipeline import Pipeline
    from lda_transformer import LDATransformer

    pipe = Pipeline([
        ("lda", LDATransformer(mode="replace")),
        ("clf", XGBClassifier()),
    ])
    pipe.fit(X_train, y_train)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


# ── Features that receive log1p before LDA ────────────────────────────────────
# DTP features are intentionally excluded: they can be negative (early payers),
# and their scale is already compact.
_LOG1P_FEATURES = [
    "credit_sale_amount",
    "opening_balance",
    "amount_due_cumsum",
    "amount_paid_cumsum",
]

# n_classes is intentionally NOT hardcoded here.  It is inferred from the
# unique values of y inside fit() so that LDATransformer works correctly
# for both the full 4-class problem AND the delinquent-only 3-class problem
# used by TwoStageClassifier Stage 2, without any caller-side configuration.


class LDATransformer(BaseEstimator, TransformerMixin):
    """
    Supervised dimensionality reducer using Linear Discriminant Analysis.

    Fits on (X_train, y_train), transforms any X to LD components.
    Optionally appends components to the original feature matrix.

    Parameters
    ----------
    n_components : int or None
        Number of discriminant components to keep.
        None (default) uses the maximum possible: min(n_classes-1, n_features).
    mode : {"append", "replace"}
        "append"  → return original X columns + new LD columns (default).
        "replace" → return only the LD columns.
    log1p_cols : list of str or None
        Columns to apply log1p before fitting/transforming.
        Defaults to _LOG1P_FEATURES.  Pass [] to disable entirely.
    verbose : bool
        If True, print a summary after fitting.

    Attributes
    ----------
    scaler_ : StandardScaler
        Fitted scaler applied before LDA.
    lda_ : LinearDiscriminantAnalysis
        Fitted LDA model.
    ld_columns_ : list of str
        Names of the output LD columns, e.g. ["LD1", "LD2", "LD3"].
    feature_names_in_ : list of str
        Column names seen during fit (used to align transform input).
    log1p_cols_ : list of str
        Subset of log1p_cols that were actually present during fit.
    n_classes_ : int
        Number of unique class labels seen during fit.  Automatically inferred
        from ``y`` so the transformer works for both the full 4-class problem
        and the delinquent-only 3-class problem in ``TwoStageClassifier``.

    Notes
    -----
    Because ``n_classes_`` is inferred at fit time, the same ``LDATransformer``
    class is used for both stages of ``TwoStageClassifier`` without any
    configuration change:

    - **Stage 1 / full dataset** — ``y`` contains {0,1,2,3} → n_classes_=4,
      max components=3, producing LD1/LD2/LD3.
    - **Stage 2 / delinquent only** — ``y`` contains {0,1,2} (remapped) →
      n_classes_=3, max components=2, producing LD1/LD2 that are optimised
      *specifically* for separating 30/60/90-day classes without on_time noise.
    """

    def __init__(
        self,
        n_components: int | None = None,
        mode: str = "append",
        log1p_cols: list[str] | None = None,
        verbose: bool = False,
    ):
        if mode not in ("append", "replace"):
            raise ValueError(f"mode must be 'append' or 'replace', got {mode!r}")
        self.n_components = n_components
        self.mode         = mode
        self.log1p_cols   = log1p_cols
        self.verbose      = verbose

    # ── internal helpers ──────────────────────────────────────────────────────

    def _resolve_log1p_cols(self, columns: list[str]) -> list[str]:
        """Return the intersection of requested log1p cols and present columns."""
        candidates = self.log1p_cols if self.log1p_cols is not None else _LOG1P_FEATURES
        return [c for c in candidates if c in columns]

    def _apply_log1p(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p (with non-negative shift) to financial columns."""
        X = X.copy()
        for col in self.log1p_cols_:
            shift = abs(X[col].min()) if X[col].min() < 0 else 0.0
            X[col] = np.log1p(X[col] + shift)
        return X

    def _to_float_df(self, X) -> pd.DataFrame:
        """Ensure X is a float64 DataFrame with consistent column names."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X.astype(float)

    # ── sklearn API ───────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y):
        """
        Fit the scaler and LDA on X, y.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
            Class labels (encoded integers or strings).

        Returns
        -------
        self

        Notes
        -----
        ``n_classes_`` is set here by inspecting ``np.unique(y)``.  This means
        the same class can be used for Stage 1 (4 classes → 3 components) and
        Stage 2 of ``TwoStageClassifier`` (3 delinquent classes → 2 components)
        without any configuration change from the caller.
        """
        X = self._to_float_df(X) if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        self.log1p_cols_       = self._resolve_log1p_cols(self.feature_names_in_)

        X_proc = self._apply_log1p(X)

        # Infer n_classes from y so this transformer works for both the
        # 4-class full-dataset problem and the 3-class delinquent-only
        # problem used by TwoStageClassifier Stage 2.
        self.n_classes_ = len(np.unique(y))
        n_components    = self.n_components or min(self.n_classes_ - 1, X_proc.shape[1])
        self.ld_columns_ = [f"LD{i+1}" for i in range(n_components)]

        self.scaler_ = StandardScaler()
        X_scaled     = self.scaler_.fit_transform(X_proc)

        self.lda_ = LinearDiscriminantAnalysis(
            n_components=n_components, solver="svd"
        )
        self.lda_.fit(X_scaled, y)

        if self.verbose:
            evr = self.lda_.explained_variance_ratio_
            print("\n── LDATransformer fitted ──────────────────────────")
            print(f"  n_classes : {self.n_classes_}  "
                  f"({'4-class full' if self.n_classes_ == 4 else f'{self.n_classes_}-class delinquent'})")
            print(f"  components: {n_components}")
            for i, v in enumerate(evr, 1):
                bar = "█" * int(v * 35)
                print(f"  LD{i}: {bar:<35}  {v*100:.1f}% sep. variance")
            print(f"  log1p applied to: {self.log1p_cols_ or 'none'}")
            print(f"  mode: {self.mode}")
            print("────────────────────────────────────────────────────\n")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Project X into LDA space.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Must contain the same columns seen during fit.

        Returns
        -------
        pd.DataFrame
            If mode="append": original columns + LD1 … LDk.
            If mode="replace": only LD1 … LDk.
        """
        check_is_fitted(self, ["scaler_", "lda_", "feature_names_in_"])

        X_orig = self._to_float_df(X).reindex(columns=self.feature_names_in_)
        X_proc = self._apply_log1p(X_orig)
        X_scaled = self.scaler_.transform(X_proc)

        ld_values = self.lda_.transform(X_scaled)
        ld_df     = pd.DataFrame(ld_values, columns=self.ld_columns_, index=X_orig.index)

        if self.mode == "replace":
            return ld_df

        # mode == "append": concatenate original + LD columns
        return pd.concat(
            [X_orig.reset_index(drop=True), ld_df.reset_index(drop=True)],
            axis=1,
        )

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        """Fit and transform in one step (required by TransformerMixin)."""
        return self.fit(X, y).transform(X)

    # ── convenience ───────────────────────────────────────────────────────────

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Explained separation variance ratio per component."""
        check_is_fitted(self, "lda_")
        return self.lda_.explained_variance_ratio_

    def coef_dataframe(self) -> pd.DataFrame:
        """
        Return LDA coefficients as a tidy DataFrame.

        Rows = features, columns = classes (from lda_.classes_).
        Larger absolute values indicate stronger contribution of that
        feature to separating that class from the rest.

        Returns
        -------
        pd.DataFrame, shape (n_features, n_classes)
        """
        check_is_fitted(self, "lda_")
        return pd.DataFrame(
            self.lda_.coef_.T,
            index=self.feature_names_in_,
            columns=[f"class_{c}" for c in self.lda_.classes_],
        )
