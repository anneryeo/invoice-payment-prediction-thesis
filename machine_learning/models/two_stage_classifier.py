import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import SelectFromModel

from .base_pipeline import BasePipeline
from machine_learning.utils.features.lda_transformer import LDATransformer


class TwoStageClassifier(BaseEstimator, ClassifierMixin):
    """
    Two-stage payment delay classifier.

    Decomposes the 4-class problem into two structurally distinct sub-problems
    that respond to different feature signals:

    - **Stage 1** — binary: on_time (0) vs any late payment (1, 2, 3).
      Trained on the full dataset. Dominated by financial-state features:
      opening_balance, payment_ratio. Strong signal (separation ~0.36–0.40).

    - **Stage 2** — multiclass: 30_days vs 60_days vs 90_days.
      Trained **only on late invoices** — on_time records are never seen.
      Dominated by DTP history features: dtp_avg, dtp_1. Weaker but cleaner
      signal (separation ~0.22–0.24) because the majority-class noise from
      on_time is removed entirely.

    Final class probabilities are recovered by the chain rule:
        P(class = k) = P(late) × P(class = k | late)    for k in {1, 2, 3}
        P(on_time)   = 1 - P(late)

    This differs fundamentally from ``OrdinalClassifier``, which also trains
    multiple binary classifiers but always on the full dataset with cumulative
    thresholds. ``TwoStageClassifier`` uses population isolation — Stage 2
    never sees the on_time population — which removes the structural noise
    those records introduce when distinguishing degree of lateness.

    Parameters
    ----------
    stage1_estimator : sklearn-compatible classifier
        Unfitted binary estimator for the on_time vs late decision.
        Any estimator with ``fit`` and ``predict_proba`` works. A clone
        is fitted internally so the original object is never mutated.
    stage2_estimator : sklearn-compatible classifier
        Unfitted multiclass estimator for 30_days / 60_days / 90_days.
        Trained only on the late-invoice subset of the training data.
        A clone is fitted internally.

    Attributes
    ----------
    stage1_estimator_ : fitted estimator
        Fitted clone of ``stage1_estimator``.
    stage2_estimator_ : fitted estimator
        Fitted clone of ``stage2_estimator``, trained on late invoices only.
    classes_ : np.ndarray of shape (4,)
        Class labels [0, 1, 2, 3] in ascending ordinal order.
    feature_importances_ : np.ndarray of shape (n_features,)
        Average of Stage 1 and Stage 2 feature importances. Only available
        when both base estimators expose ``feature_importances_``.
    stage2_lda_ : LDATransformer or None
        Fitted delinquent-only LDA transformer applied to Stage 2's training
        and inference data.  ``None`` when ``use_lda=False``.  Fitted on the
        late-invoice rows only (y > 0) so its components are optimised for
        separating 30/60/90-day classes without on_time noise.  Uses
        ``mode="append"`` by default so Stage 2 retains all original features
        alongside the compressed LD1/LD2 signal.

    Notes
    -----
    Stage 2 requires at least one sample from each late class in the training
    set. If a class is absent (e.g. no 60_days invoices after filtering),
    Stage 2 will raise a ``ValueError`` from the underlying estimator.

    When ``use_lda=True``, Stage 2 receives a dedicated ``LDATransformer``
    fitted *only* on the delinquent population.  This is distinct from any
    LDA applied upstream in ``run_models_parallel.py``, which is a 4-class
    transformer fitted on the full dataset for Stage 1's benefit.  The two
    transformers are independent and produce different discriminant axes:

    - **Full-dataset LDA** (upstream, 4 classes) → LD1 mainly captures
      on_time vs everyone else (≈97% of separation variance).
    - **Delinquent-only LDA** (here, 3 classes) → LD1/LD2 are fully
      dedicated to separating 30 / 60 / 90 days from each other.
    """

    @staticmethod
    def _to_named_df(X, reference_names=None):
        """
        Convert X to a DataFrame with all-string column names.

        When X arrives as a numpy array (e.g. after np.asarray() or boolean
        masking), column indices are integers.  LDATransformer appends string
        columns (LD1, LD2 …) which produces a mixed int/str column set that
        sklearn estimators reject with a TypeError.

        This helper guarantees all column names are strings before the data
        enters LDATransformer, and is the single point where that conversion
        is enforced for both fit() and fit_with_masks().

        Parameters
        ----------
        X : array-like or pd.DataFrame
        reference_names : list of str or None
            If provided, use these as column names (e.g. from a previously
            fitted LDATransformer.feature_names_in_).  Otherwise names are
            generated as "f0", "f1", … from the column count.

        Returns
        -------
        pd.DataFrame with dtype float64 and all-string column names.
        """
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X.columns = X.columns.astype(str)
            return X.astype(float)

        # numpy array — build column names
        n_cols = X.shape[1]
        if reference_names is not None and len(reference_names) == n_cols:
            cols = [str(c) for c in reference_names]
        else:
            cols = [f"f{i}" for i in range(n_cols)]
        return pd.DataFrame(X, columns=cols, dtype=float)

    def __init__(self, stage1_estimator, stage2_estimator, use_lda: bool = False,
                 lda_mode: str = "append"):
        self.stage1_estimator = stage1_estimator
        self.stage2_estimator = stage2_estimator
        self.use_lda          = use_lda
        self.lda_mode         = lda_mode

    def fit(self, X, y):
        """
        Fit both stages on their respective training populations.

        Stage 1 is trained on the full (X, y) with y binarised to
        on_time=0 vs late=1.  Stage 2 is trained only on the rows where
        y > 0, preserving the original late-class labels (1, 2, 3).

        When ``use_lda=True``, a dedicated ``LDATransformer`` is fitted on the
        delinquent subset *before* Stage 2's estimator sees the data.  This
        transformer (``stage2_lda_``) is stored and applied automatically at
        inference time in ``predict_proba``.  It is independent of any upstream
        LDA applied to the full dataset — its components are optimised solely
        for separating 30/60/90-day classes without on_time noise.

        Both stages receive the full feature matrix by default.  To train each
        stage on its own selected feature subset, call ``fit_with_masks``
        instead, which is invoked automatically by ``TwoStagePipeline`` when
        ``use_feature_selection=True``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Integer-encoded ordinal labels: on_time=0, 30_days=1,
            60_days=2, 90_days=3.

        Returns
        -------
        self : TwoStageClassifier
        """
        y = np.asarray(y)

        # ── Stage 1 — full dataset, binary target ─────────────────────────────
        # y is binarised: on_time=0, any late class=1.
        y_binary = (y > 0).astype(int)
        self.stage1_estimator_ = clone(self.stage1_estimator)
        self.stage1_estimator_.fit(X, y_binary)

        # ── Stage 2 — late invoices only ──────────────────────────────────────
        # Only rows where y > 0 (30/60/90-day) are used.  Labels are remapped
        # {1,2,3} → {0,1,2} because XGBoost and sklearn require contiguous
        # zero-based class indices.  stage2_offset_=1 is stored so that
        # predict_proba can add it back when writing into the 4-class output.
        late_mask = y > 0
        X_late    = X[late_mask]
        y_late    = y[late_mask] - 1  # remap {1,2,3} → {0,1,2}

        # Optional delinquent-only LDA.
        # Fitted here on (X_late, y_late) so it sees the same population that
        # Stage 2's estimator trains on.  n_classes_ is inferred as 3 from
        # y_late, giving 2 discriminant components (LD1/LD2) that are fully
        # dedicated to the 30 vs 60 vs 90-day boundary.
        if self.use_lda:
            # Ensure all column names are strings before LDA appends LD1/LD2.
            # np.asarray() earlier strips column names, leaving integer indices
            # that mix with the new string LD columns and cause a TypeError in
            # sklearn estimators.
            X_late_df = self._to_named_df(X_late)
            self.stage2_lda_ = LDATransformer(mode=self.lda_mode, verbose=False)
            # fit_transform returns a DataFrame; convert to numpy so the
            # downstream estimator receives a consistent array type.
            X_late = self.stage2_lda_.fit_transform(X_late_df, y_late).to_numpy()
        else:
            self.stage2_lda_ = None

        self.stage2_estimator_ = clone(self.stage2_estimator)
        self.stage2_estimator_.fit(X_late, y_late)

        self.classes_       = np.array([0, 1, 2, 3])
        self.stage2_offset_ = 1   # added back in predict_proba to recover {1,2,3}
        self.mask1_         = None   # set by fit_with_masks; None means use full X
        self.mask2_         = None
        return self

    def fit_with_masks(self, X, y, mask1, mask2):
        """
        Fit both stages using their own independent feature subsets.

        Called by ``TwoStagePipeline`` when ``use_feature_selection=True``
        after the initial full fit has been used to derive per-stage importance
        masks.  Each stage is refitted on only the features that are informative
        for its specific decision boundary:

        - Stage 1 uses ``mask1``, derived from Stage 1's own
          ``feature_importances_`` on the binary on_time vs late problem.
          Financial-state features (opening_balance, payment_ratio) dominate.

        - Stage 2 uses ``mask2``, derived from Stage 2's own
          ``feature_importances_`` on the late-only multiclass problem.
          DTP history features (dtp_avg, dtp_1) dominate.

        When ``use_lda=True``, the delinquent-only LDA is re-fitted on the
        masked feature subset ``X[late_mask][:, mask2]`` — not on the full
        feature matrix — so the discriminant axes reflect the same reduced
        feature space that Stage 2's estimator trains on.  ``stage2_lda_`` is
        updated in-place and ``predict_proba`` applies the new transformer
        automatically.

        ``predict_proba`` automatically applies the stored masks (and the LDA
        transformer when present) at inference time.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Full feature matrix — masking is applied internally.
        y : array-like of shape (n_samples,)
            Integer-encoded ordinal labels: on_time=0, 30_days=1,
            60_days=2, 90_days=3.
        mask1 : np.ndarray of bool, shape (n_features,)
            Boolean mask selecting features for Stage 1.
        mask2 : np.ndarray of bool, shape (n_features,)
            Boolean mask selecting features for Stage 2 (applied before LDA).

        Returns
        -------
        self : TwoStageClassifier
        """
        X = np.asarray(X)
        y = np.asarray(y)

        y_binary  = (y > 0).astype(int)
        late_mask = y > 0

        # ── Stage 1 — masked feature subset, binary target ────────────────────
        self.stage1_estimator_ = clone(self.stage1_estimator)
        self.stage1_estimator_.fit(X[:, mask1], y_binary)

        # ── Stage 2 — masked late-invoice subset ──────────────────────────────
        # Apply mask2 first to get the feature-selected late subset, then
        # optionally project through the delinquent-only LDA before fitting.
        # Remap {1,2,3} → {0,1,2} to satisfy zero-based class index requirement.
        X_late_masked = X[late_mask][:, mask2]
        y_late        = y[late_mask] - 1  # remap {1,2,3} → {0,1,2}

        if self.use_lda:
            # Re-fit LDA on the masked subset so its axes reflect the reduced
            # feature space.  Convert to a named DataFrame first (same reason
            # as in fit() — boolean masking of a numpy array leaves integer
            # column indices that conflict with the string LD columns).
            X_late_masked_df = self._to_named_df(X_late_masked)
            self.stage2_lda_ = LDATransformer(mode=self.lda_mode, verbose=False)
            X_late_masked = self.stage2_lda_.fit_transform(
                X_late_masked_df, y_late
            ).to_numpy()
        else:
            self.stage2_lda_ = None

        self.stage2_estimator_ = clone(self.stage2_estimator)
        self.stage2_estimator_.fit(X_late_masked, y_late)

        self.classes_       = np.array([0, 1, 2, 3])
        self.stage2_offset_ = 1
        self.mask1_         = mask1
        self.mask2_         = mask2
        return self

    def predict_proba(self, X):
        """
        Estimate class probabilities via the two-stage chain rule.

        Computes P(on_time), P(30_days), P(60_days), P(90_days) as:
            P(on_time)   = 1 - P(late)
            P(k | k > 0) = P(late) * P(class=k | late)

        When ``fit_with_masks`` was used, each stage receives only the
        feature columns it was trained on (``mask1_`` for Stage 1,
        ``mask2_`` for Stage 2). When ``fit`` was used without masks,
        both stages receive the full feature matrix.

        Stage 2 columns are aligned to the late classes (1, 2, 3) using
        ``stage2_estimator_.classes_`` so that column order is robust to
        how the estimator internally sorts its class labels.

        When ``use_lda=True``, the stored ``stage2_lda_`` transformer is applied
        to X2 (after masking) before Stage 2's estimator produces probabilities.
        The transformer was fitted on the same feature-selected delinquent
        population at training time, so inference exactly mirrors training.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Full feature matrix.  Internal masking is applied automatically
            when per-stage masks were set during ``fit_with_masks``, and the
            delinquent-only LDA is applied when ``use_lda=True``.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, 4)
            Row-normalised class probability estimates.
        """
        X = np.asarray(X)
        n = X.shape[0]

        # ── Feature masking ───────────────────────────────────────────────────
        # Apply per-stage feature masks when set by fit_with_masks;
        # fall back to the full feature matrix when fit() was used without masks.
        X1 = X[:, self.mask1_] if self.mask1_ is not None else X
        X2 = X[:, self.mask2_] if self.mask2_ is not None else X

        # ── Delinquent-only LDA (Stage 2 only) ───────────────────────────────
        # If the classifier was trained with use_lda=True, project X2 through
        # the stored delinquent-only transformer before Stage 2 scores it.
        # This mirrors the transform applied to X_late during fit/fit_with_masks.
        if self.stage2_lda_ is not None:
            # Reconstruct a properly-named DataFrame using the column names
            # recorded at fit time, then project through the stored LDA and
            # convert back to numpy for the downstream estimator.
            X2 = self.stage2_lda_.transform(
                self._to_named_df(X2, self.stage2_lda_.feature_names_in_)
            ).to_numpy()

        p_late           = self.stage1_estimator_.predict_proba(X1)[:, 1]
        p_given_late_raw = self.stage2_estimator_.predict_proba(X2)

        # Map stage 2 columns to positions 1, 2, 3 using the stored offset.
        # stage2_estimator_.classes_ is [0,1,2] (zero-based); adding stage2_offset_
        # (always 1) recovers the original late-class indices {1,2,3} so each
        # probability lands in the correct column of the 4-class output array.
        proba = np.zeros((n, 4))
        proba[:, 0] = 1.0 - p_late

        for col_idx, class_label in enumerate(self.stage2_estimator_.classes_):
            proba[:, int(class_label) + self.stage2_offset_] = p_late * p_given_late_raw[:, col_idx]

        # Normalise rows to absorb any floating-point drift
        row_sums = proba.sum(axis=1, keepdims=True)
        proba    = proba / np.where(row_sums == 0, 1.0, row_sums)

        return proba

    def predict(self, X):
        """
        Predict the most probable payment class for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted integer class labels (0, 1, 2, or 3).
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Average feature importances across both stages.

        Used by ``TwoStagePipeline`` during the initial full fit to derive
        independent per-stage importance masks via separate ``SelectFromModel``
        calls. After that first fit, the pipeline calls ``fit_with_masks``
        so each stage is retrained on only the features that matter for its
        specific decision. The average here is therefore only used once —
        to bootstrap the per-stage selectors — not as the final selection.

        Returns
        -------
        importances : np.ndarray of shape (n_features,)

        Raises
        ------
        AttributeError
            If either base estimator does not expose ``feature_importances_``.
            Tree-based models (XGBoost, Random Forest, AdaBoost, Decision Tree)
            support this. KNN, MLP, and Gaussian NB do not.
            Also raised if called after ``fit_with_masks`` when stage 1 and
            stage 2 were trained on different-sized feature subsets.
        """
        for stage, est in [("Stage 1", self.stage1_estimator_),
                            ("Stage 2", self.stage2_estimator_)]:
            if not hasattr(est, "feature_importances_"):
                raise AttributeError(
                    f"{stage} estimator ({type(est).__name__}) does not expose "
                    "feature_importances_. Use a tree-based estimator for both "
                    "stages, or set use_feature_selection=False."
                )
        fi1 = self.stage1_estimator_.feature_importances_
        fi2 = self.stage2_estimator_.feature_importances_
        if fi1.shape != fi2.shape:
            raise AttributeError(
                "feature_importances_ is not available after fit_with_masks "
                "when stage 1 and stage 2 have different feature subsets "
                f"(stage1: {fi1.shape[0]} features, stage2: {fi2.shape[0]} features)."
            )
        return (fi1 + fi2) / 2.0



class TwoStagePipeline(BasePipeline):
    """
    Pipeline wrapper for two-stage payment delay classification.

    Wraps ``TwoStageClassifier`` so it fits the ``BasePipeline`` interface,
    giving it access to feature selection, evaluation, and result logging
    exactly as the other pipeline classes do. The call pattern is identical
    to ``OrdinalPipeline`` and ``AdaBoostPipeline``.

    The two stages use the same feature set by default. If you want to pass
    different features to each stage (e.g. financial features to Stage 1,
    DTP features to Stage 2), do the column selection upstream in
    ``DataPreparer`` before passing the data in — the pipeline itself does
    not split features between stages.

    Parameters
    ----------
    X_train, X_test, y_train, y_test : array-like
        Train/test splits passed through to ``BasePipeline``.
    args : any
        Forwarded to ``BasePipeline``.
    parameters : dict, optional
        Accepted keys (all optional):

        ``scale_pos_weight`` : bool, default False
            Not used by ``TwoStageClassifier`` directly — included for
            consistency with ``OrdinalPipeline`` parameter conventions.
            Pass class weights to the individual estimators instead.

    feature_names : list of str, optional
        Column names for feature importance logging.
    stage1_estimator : sklearn-compatible classifier
        Unfitted estimator for Stage 1 (on_time vs late).
        Must expose ``predict_proba``. Should be a tree-based model
        if ``use_feature_selection=True`` is intended.
    stage2_estimator : sklearn-compatible classifier
        Unfitted estimator for Stage 2 (30_days / 60_days / 90_days,
        trained on late invoices only). Must expose ``predict_proba``.
    use_lda : bool, default False
        When True, a dedicated ``LDATransformer`` is fitted on the late-invoice
        training rows (y > 0) and applied to Stage 2's input at both training
        and inference time.  The transformer is independent of any upstream LDA
        applied to the full dataset — its 2 discriminant components (LD1/LD2)
        are optimised specifically for separating 30 / 60 / 90-day classes.
    lda_mode : {"append", "replace"}, default "append"
        Passed to the internal ``LDATransformer``.  "append" keeps all original
        Stage 2 features and adds LD1/LD2; "replace" uses only LD1/LD2.
        "append" is recommended for tree-based Stage 2 estimators.

    Examples
    --------
    >>> from xgboost import XGBClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> pipeline = TwoStagePipeline(
    ...     X_train, X_test, y_train, y_test, args,
    ...     stage1_estimator=XGBClassifier(
    ...         max_depth=3, learning_rate=0.01, n_estimators=500,
    ...     ),
    ...     stage2_estimator=RandomForestClassifier(
    ...         max_depth=10, n_estimators=200,
    ...     ),
    ... )
    >>> pipeline.initialize_model().fit(use_feature_selection=True).evaluate()

    Notes
    -----
    ``use_feature_selection=True`` requires both estimators to expose
    ``feature_importances_`` (tree-based models only). Each stage gets its
    own ``SelectFromModel`` selector built from its own importances, so
    Stage 1 retains the financial-state features that drive the binary
    on_time vs late boundary, while Stage 2 retains the DTP history features
    that drive the degree-of-lateness boundary. The union of both masks is
    logged to ``_set_features`` for result tracking. ``predict_proba`` applies
    the stored per-stage masks automatically at inference time.
    """

    def __init__(
        self,
        X_train, X_test,
        y_train, y_test,
        args,
        parameters=None,
        feature_names=None,
        stage1_estimator=None,
        stage2_estimator=None,
        use_lda: bool = False,
        lda_mode: str = "append",
    ):
        super().__init__(X_train, X_test, y_train, y_test, args, parameters, feature_names)

        if stage1_estimator is None or stage2_estimator is None:
            raise ValueError(
                "Both stage1_estimator and stage2_estimator must be provided. "
                "Pass pre-instantiated sklearn-compatible classifiers."
            )

        self._stage1_estimator = stage1_estimator
        self._stage2_estimator = stage2_estimator
        # use_lda / lda_mode are forwarded to TwoStageClassifier.initialize_model()
        # so the delinquent-only LDA is wired into Stage 2 before any fitting occurs.
        self._use_lda  = use_lda
        self._lda_mode = lda_mode

    def initialize_model(self):
        """
        Construct the ``TwoStageClassifier`` and assign it to ``self.model``.

        Forwards ``use_lda`` and ``lda_mode`` to ``TwoStageClassifier`` so that
        the delinquent-only LDA is configured before any fitting occurs.  When
        ``use_lda=True``, ``TwoStageClassifier.fit()`` will fit a dedicated
        ``LDATransformer`` on the late-invoice training rows and store it as
        ``stage2_lda_`` for use at inference time in ``predict_proba``.

        Returns
        -------
        self : TwoStagePipeline
        """
        self.model = TwoStageClassifier(
            stage1_estimator=self._stage1_estimator,
            stage2_estimator=self._stage2_estimator,
            use_lda=self._use_lda,
            lda_mode=self._lda_mode,
        )
        return self

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_stage_masks(self, threshold):
        """
        Build per-stage boolean feature-selection masks from the importances
        of the already-fitted ``self.model``.

        Stage 1's mask is derived via ``SelectFromModel`` on the binary
        estimator's importances.  Stage 2's mask replicates SelectFromModel's
        threshold logic directly on the sliced importance vector so that any
        appended LDA columns are excluded before the shape is compared with
        Stage 1's mask.

        When ``use_lda=True`` and ``lda_mode="replace"``, Stage 2's estimator
        was trained on LD columns only — its importances cannot be aligned with
        the original feature space — so all original features are selected for
        Stage 2 (``mask2`` is all-True) and ``fit_with_masks`` re-derives the
        LD columns internally.

        Parameters
        ----------
        threshold : str or float
            Importance threshold passed to ``SelectFromModel`` for Stage 1 and
            replicated manually for Stage 2.  Accepts ``"median"``, ``"mean"``,
            or a numeric value.

        Returns
        -------
        mask1 : np.ndarray of bool, shape (n_original,)
            Features selected for Stage 1.
        mask2 : np.ndarray of bool, shape (n_original,)
            Features selected for Stage 2.
        """
        n_original = self.X_train.shape[1]

        # Stage 1 — always trained on the full original feature space.
        sel1  = SelectFromModel(
            self.model.stage1_estimator_, threshold=threshold, prefit=True
        )
        mask1 = sel1.get_support()

        # Stage 2 — skip SelectFromModel to avoid shape issues from LDA columns.
        if self._use_lda and self._lda_mode == "replace":
            mask2 = np.ones(n_original, dtype=bool)
        else:
            fi2 = self.model.stage2_estimator_.feature_importances_[:n_original]
            if threshold == "median":
                thresh_val = np.median(fi2)
            elif threshold == "mean":
                thresh_val = np.mean(fi2)
            else:
                thresh_val = float(threshold)
            mask2 = fi2 >= thresh_val

        return mask1, mask2

    def _log_feature_selection(self, mask1, mask2, threshold):
        """
        Record feature-selection metadata to ``self.features`` via
        ``_set_features``.

        Logs the **union** of both stage masks so the result row captures every
        feature used by either stage.  Stage 1's full-fit importances are used
        as the importance scores for the union because Stage 1 always trains on
        the original feature space and provides a consistent n_original-length
        importance vector.

        Parameters
        ----------
        mask1 : np.ndarray of bool, shape (n_original,)
        mask2 : np.ndarray of bool, shape (n_original,)
        threshold : str or float
            The threshold value used to build both masks; recorded in the
            parameters string for traceability.
        """
        union_mask = mask1 | mask2
        lda_suffix = " + Stage-2 delinquent LDA" if self._use_lda else ""
        self._set_features(
            method_text       = "Two-stage independent feature selection" + lda_suffix,
            method_parameters = f"threshold={threshold!r}, "
                                f"stage1_features={int(mask1.sum())}, "
                                f"stage2_features={int(mask2.sum())}",
            mask              = union_mask,
            importances       = self.model.stage1_estimator_.feature_importances_,
        )

    def _write_per_stage_weights(self, mask1=None, mask2=None):
        """
        Overwrite ``self.features.weights`` with a nested dict
        ``{"stage_1": {feat: score, …}, "stage_2": {feat: score, …}}``
        so ``FeatureInfo.__repr__`` renders both stage blocks.

        Must be called **after** the final ``fit`` or ``fit_with_masks`` call
        so the importances reflect the estimators that will actually be used at
        inference time.

        When ``use_feature_selection=True`` (``mask1``/``mask2`` provided):
            Each estimator was re-fitted on its own masked feature subset, so
            its ``feature_importances_`` has shape ``(mask.sum(),)``.  The
            scores are padded back to the full original feature space before
            keying into the dict, keeping only the selected entries.

        When ``use_feature_selection=False`` (masks are ``None``):
            Both estimators were fitted on the full feature matrix, so their
            importances already have shape ``(n_original,)``.  No padding is
            needed.

        When ``lda_mode="replace"``, Stage 2's estimator was trained on LD
        columns only.  Its ``feature_importances_`` describes LD axes, not
        original features, so Stage 2 scores are set to zero rather than
        producing a misleading mapping.

        Parameters
        ----------
        mask1 : np.ndarray of bool or None
            Boolean mask used to select Stage 1 features.  ``None`` when no
            feature selection was applied.
        mask2 : np.ndarray of bool or None
            Boolean mask used to select Stage 2 features.  ``None`` when no
            feature selection was applied.
        """
        _names = self.original_feature_names
        if _names is None:
            return

        n_original   = self.X_train.shape[1]
        _lda_replace = self._use_lda and self._lda_mode == "replace"

        if mask1 is not None and mask2 is not None:
            # ── Feature-selection path: estimators trained on masked subsets ──
            # Stage 1: importances shape (mask1.sum(),) — pad to (n_original,).
            fi1 = np.zeros(n_original)
            fi1[mask1] = self.model.stage1_estimator_.feature_importances_

            # Stage 2: build a {feature_name: importance} dict directly rather
            # than padding into a fixed-length array, because lda_mode="append"
            # adds LD columns that have no slot in the original feature space.
            #
            # lda_mode="replace" → estimator trained on LD columns only;
            #   importances describe LD axes, not original features.  Record
            #   only the LD column scores (no original-feature entries).
            # lda_mode="append" → estimator trained on [mask2 originals | LDs];
            #   first mask2.sum() importances map to original features,
            #   remaining importances map to LD1, LD2, … columns.
            # no LDA → importances map 1-to-1 with mask2 original features.
            stage2_weights: dict = {}
            if hasattr(self.model.stage2_estimator_, "feature_importances_"):
                fi2_full = self.model.stage2_estimator_.feature_importances_
                lda_     = self.model.stage2_lda_   # None when use_lda=False

                if _lda_replace:
                    # LD columns only — derive names as LD1, LD2, …
                    n_ld = len(fi2_full)
                    ld_names = [f"LD{i+1}" for i in range(n_ld)]
                    stage2_weights = {
                        ld: round(float(s), 6)
                        for ld, s in zip(ld_names, fi2_full)
                    }
                else:
                    # Original features (first mask2.sum() entries)
                    orig_names = [n for n, keep in zip(_names, mask2) if keep]
                    for name, score in zip(orig_names, fi2_full[:mask2.sum()]):
                        stage2_weights[name] = round(float(score), 6)

                    # Appended LD columns (remaining entries, only when use_lda=True)
                    if lda_ is not None:
                        n_ld     = len(fi2_full) - mask2.sum()
                        ld_names = [f"LD{i+1}" for i in range(n_ld)]
                        for ld, score in zip(ld_names, fi2_full[mask2.sum():]):
                            stage2_weights[ld] = round(float(score), 6)

            self.features.weights = {
                "stage_1": {
                    n: round(float(s), 6)
                    for n, s, keep in zip(_names, fi1, mask1) if keep
                },
                "stage_2": stage2_weights,
            }
        else:
            # ── No feature selection: estimators trained on full matrix ───────
            fi1 = self.model.stage1_estimator_.feature_importances_

            stage2_weights: dict = {}
            if hasattr(self.model.stage2_estimator_, "feature_importances_"):
                fi2_full = self.model.stage2_estimator_.feature_importances_
                lda_     = self.model.stage2_lda_

                if _lda_replace:
                    # LD columns only
                    n_ld     = len(fi2_full)
                    ld_names = [f"LD{i+1}" for i in range(n_ld)]
                    stage2_weights = {
                        ld: round(float(s), 6)
                        for ld, s in zip(ld_names, fi2_full)
                    }
                else:
                    # Original features
                    for name, score in zip(_names, fi2_full[:n_original]):
                        stage2_weights[name] = round(float(score), 6)
                    # Appended LD columns
                    if lda_ is not None:
                        n_ld     = len(fi2_full) - n_original
                        ld_names = [f"LD{i+1}" for i in range(n_ld)]
                        for ld, score in zip(ld_names, fi2_full[n_original:]):
                            stage2_weights[ld] = round(float(score), 6)

            self.features.weights = {
                "stage_1": {n: round(float(s), 6) for n, s in zip(_names, fi1)},
                "stage_2": stage2_weights,
            }

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, use_feature_selection=False, threshold="median"):
        """
        Train the two-stage model, optionally applying independent per-stage
        feature selection.

        When ``use_feature_selection=True`` the procedure is:

        1. Fit the full ``TwoStageClassifier`` on all features so both
           stages have trained estimators from which importances can be read.
        2. Call ``_build_stage_masks`` to derive ``mask1`` and ``mask2`` from
           each stage's own importances independently.
        3. Call ``_log_feature_selection`` to record the union mask and
           feature metadata to ``self.features`` via ``_set_features``.
        4. Call ``fit_with_masks`` to retrain each stage on its own feature
           subset.  ``predict_proba`` then applies the stored masks automatically.
        5. Call ``_write_per_stage_weights`` to overwrite the flat weights with
           a nested ``{"stage_1": …, "stage_2": …}`` dict.

        When ``use_feature_selection=False``, both stages use the full feature
        matrix, ``_set_features`` records ``method_text="none"``, and
        ``_write_per_stage_weights`` is still called so the per-stage breakdown
        is always available.

        Parameters
        ----------
        use_feature_selection : bool, default False
            Whether to apply per-stage importance-based feature selection.
            Requires both stage estimators to expose ``feature_importances_``
            (tree-based models only).
        threshold : str or float, default "median"
            Passed to both ``SelectFromModel`` instances independently.
            Stage 1 and Stage 2 may select different numbers of features
            even with the same threshold because their importance
            distributions differ.

        Returns
        -------
        self : TwoStagePipeline

        Raises
        ------
        ValueError
            If ``initialize_model()`` has not been called prior to ``fit()``.
        AttributeError
            If ``use_feature_selection=True`` but either stage estimator does
            not expose ``feature_importances_``.
        """
        if self.model is None:
            raise ValueError("Model not built. Call initialize_model() first.")

        # Initial full fit — both stages see all features so their importances
        # can be read independently to build per-stage selection masks.
        self.model.fit(self.X_train, self.y_train)

        if use_feature_selection:
            mask1, mask2 = self._build_stage_masks(threshold)
            self._log_feature_selection(mask1, mask2, threshold)
            self.model.fit_with_masks(self.X_train, self.y_train, mask1, mask2)
            self._write_per_stage_weights(mask1=mask1, mask2=mask2)
        else:
            self._set_features(method_text="none")
            self._write_per_stage_weights()

        return self
