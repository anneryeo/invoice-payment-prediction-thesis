import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import SelectFromModel

from .base_pipeline import BasePipeline


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

    Notes
    -----
    Stage 2 requires at least one sample from each late class in the training
    set. If a class is absent (e.g. no 60_days invoices after filtering),
    Stage 2 will raise a ``ValueError`` from the underlying estimator.
    """

    def __init__(self, stage1_estimator, stage2_estimator):
        self.stage1_estimator = stage1_estimator
        self.stage2_estimator = stage2_estimator

    def fit(self, X, y):
        """
        Fit both stages on their respective training populations.

        Stage 1 is trained on the full (X, y) with y binarised to
        on_time=0 vs late=1. Stage 2 is trained only on the rows where
        y > 0, preserving the original late-class labels (1, 2, 3).

        Both stages receive the full feature matrix. To train each stage
        on its own selected feature subset, call ``fit_with_masks`` instead,
        which is invoked automatically by ``TwoStagePipeline`` when
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

        # Stage 1 — full dataset, binary target
        y_binary = (y > 0).astype(int)
        self.stage1_estimator_ = clone(self.stage1_estimator)
        self.stage1_estimator_.fit(X, y_binary)

        # Stage 2 — late invoices only.
        # Labels are remapped {1,2,3} -> {0,1,2} because XGBoost and sklearn
        # require contiguous zero-based class indices.  We record the offset (1)
        # in self.stage2_offset_ so that predict_proba can add it back when
        # indexing into the 4-class output array.
        late_mask = y > 0
        X_late    = X[late_mask]
        y_late    = y[late_mask] - 1  # remap {1,2,3} -> {0,1,2}

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
        after the initial full fit has been used to derive per-stage
        importance masks. Each stage is refitted on only the features that
        are informative for its specific decision boundary:

        - Stage 1 uses ``mask1``, derived from Stage 1's own
          ``feature_importances_`` on the binary on_time vs late problem.
          Financial-state features (opening_balance, payment_ratio) dominate.

        - Stage 2 uses ``mask2``, derived from Stage 2's own
          ``feature_importances_`` on the late-only multiclass problem.
          DTP history features (dtp_avg, dtp_1) dominate.

        ``predict_proba`` automatically applies the stored masks at inference
        time so the correct feature columns are passed to each stage.

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
            Boolean mask selecting features for Stage 2.

        Returns
        -------
        self : TwoStageClassifier
        """
        X = np.asarray(X)
        y = np.asarray(y)

        y_binary  = (y > 0).astype(int)
        late_mask = y > 0

        self.stage1_estimator_ = clone(self.stage1_estimator)
        self.stage1_estimator_.fit(X[:, mask1], y_binary)

        self.stage2_estimator_ = clone(self.stage2_estimator)
        # Remap {1,2,3} -> {0,1,2} to satisfy XGBoost/sklearn zero-based requirement.
        # stage2_offset_ is stored so predict_proba can add it back when writing
        # into the correct columns of the 4-class output array.
        self.stage2_estimator_.fit(X[late_mask][:, mask2], y[late_mask] - 1)

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

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Full feature matrix. Internal masking is applied automatically
            when per-stage masks were set during ``fit_with_masks``.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, 4)
            Row-normalised class probability estimates.
        """
        X = np.asarray(X)
        n = X.shape[0]

        # Apply per-stage feature masks when set; fall back to full X otherwise
        X1 = X[:, self.mask1_] if self.mask1_ is not None else X
        X2 = X[:, self.mask2_] if self.mask2_ is not None else X

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
    ):
        super().__init__(X_train, X_test, y_train, y_test, args, parameters, feature_names)

        if stage1_estimator is None or stage2_estimator is None:
            raise ValueError(
                "Both stage1_estimator and stage2_estimator must be provided. "
                "Pass pre-instantiated sklearn-compatible classifiers."
            )

        self._stage1_estimator = stage1_estimator
        self._stage2_estimator = stage2_estimator

    def initialize_model(self):
        """
        Construct the ``TwoStageClassifier`` and assign it to ``self.model``.

        Returns
        -------
        self : TwoStagePipeline
        """
        self.model = TwoStageClassifier(
            stage1_estimator=self._stage1_estimator,
            stage2_estimator=self._stage2_estimator,
        )
        return self

    def fit(self, use_feature_selection=False, threshold="median"):
        """
        Train the two-stage model, optionally applying independent per-stage
        feature selection.

        When ``use_feature_selection=True`` the procedure is:

        1. Fit the full ``TwoStageClassifier`` on all features so both
           stages have trained estimators from which importances can be read.
        2. Build a ``SelectFromModel`` from Stage 1's own importances and a
           separate one from Stage 2's own importances. Each selector is
           applied to the same full ``X_train`` / ``X_test``, producing two
           boolean masks — ``mask1`` for Stage 1 and ``mask2`` for Stage 2.
        3. Log the **union** of both masks to ``_set_features`` so the result
           row records every feature used by either stage.
        4. Call ``fit_with_masks`` on the classifier to retrain each stage
           on its own feature subset. ``predict_proba`` then automatically
           applies the stored masks at inference time.

        When ``use_feature_selection=False``, both stages use the full
        feature matrix and ``_set_features`` records ``method_text="none"``.

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
            # Stage 1 selector — built from the binary on_time vs late importances.
            # Financial-state features (opening_balance, payment_ratio) dominate.
            sel1  = SelectFromModel(
                self.model.stage1_estimator_, threshold=threshold, prefit=True
            )
            mask1 = sel1.get_support()

            # Stage 2 selector — built from the late-only multiclass importances.
            # DTP history features (dtp_avg, dtp_1) dominate.
            sel2  = SelectFromModel(
                self.model.stage2_estimator_, threshold=threshold, prefit=True
            )
            mask2 = sel2.get_support()

            # Log the union mask — every feature used by either stage is recorded.
            # This keeps _set_features consistent with BasePipeline expectations
            # (one mask over the original feature space) while preserving the
            # full picture of what each stage selected.
            union_mask = mask1 | mask2

            # Capture importances against the union mask before refit.
            # Use Stage 1 importances for logging since Stage 1 fires on every
            # prediction; Stage 2 importances are available via
            # self.model.stage2_estimator_.feature_importances_ if needed.
            self._set_features(
                method_text       = "Two-stage independent feature selection",
                method_parameters = f"threshold={threshold!r}, "
                                    f"stage1_features={int(mask1.sum())}, "
                                    f"stage2_features={int(mask2.sum())}",
                mask              = union_mask,
                importances       = self.model.stage1_estimator_.feature_importances_,
            )

            # Retrain each stage on its own selected feature subset.
            # predict_proba applies mask1_ / mask2_ automatically at inference.
            self.model.fit_with_masks(self.X_train, self.y_train, mask1, mask2)
        else:
            self._set_features(method_text="none")

        return self
