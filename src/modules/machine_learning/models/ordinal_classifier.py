import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import SelectFromModel

from .base_pipeline import BasePipeline


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """
    Frank & Hall (2001) ordinal decomposition for multi-class classification.

    Decomposes a K-class ordinal problem into K-1 binary classifiers, where
    classifier k learns P(y > k). Final class probabilities are recovered by
    differencing the cumulative probability estimates.

    This wrapper satisfies the sklearn estimator interface so it can be used
    as a drop-in replacement for any sklearn-compatible classifier, including
    as the ``self.model`` inside a ``BasePipeline`` subclass.

    Parameters
    ----------
    base_estimator : sklearn-compatible classifier
        An unfitted estimator that implements ``fit``, ``predict_proba``, and
        optionally ``feature_importances_``. A fresh clone is created for each
        binary sub-problem so the original object is never mutated.
    scale_pos_weight : bool, default True
        If True and the base estimator accepts a ``scale_pos_weight`` parameter
        (e.g. XGBClassifier), each binary classifier's class imbalance ratio is
        computed and applied automatically. Has no effect on estimators that do
        not expose this parameter.

    Attributes
    ----------
    classifiers_ : dict of {int: fitted estimator}
        Fitted binary classifiers keyed by threshold k (0 to K-2).
        Classifier k predicts P(y > k).
    classes_ : np.ndarray
        Original class labels in ascending ordinal order, as encoded by the
        internal LabelEncoder.
    n_classes_ : int
        Number of distinct classes K.
    feature_importances_ : np.ndarray of shape (n_features,)
        Mean feature importances averaged across all K-1 binary classifiers.
        Only available after ``fit`` if the base estimator exposes
        ``feature_importances_``. Accessing this attribute on an estimator
        that does not support it raises ``AttributeError``.

    Notes
    -----
    Monotonicity of cumulative probabilities is enforced by clipping:
    P(y > k) is capped at P(y > k-1) so that class probabilities are
    always non-negative. Row sums are subsequently normalised to 1.0 to
    absorb any floating-point drift introduced by this clipping.

    References
    ----------
    Frank, E. & Hall, M. (2001). A simple approach to ordinal classification.
    Proceedings of the 12th European Conference on Machine Learning (ECML).
    """

    def __init__(self, base_estimator, scale_pos_weight=True):
        self.base_estimator    = base_estimator
        self.scale_pos_weight  = scale_pos_weight

    def fit(self, X, y):
        """
        Fit K-1 binary classifiers on the ordinal decomposition of y.

        For each threshold k in {0, …, K-2}, transforms y into a binary
        target where 1 means the original label is strictly greater than k,
        then fits a cloned copy of ``base_estimator`` on that target.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Ordinal class labels. Labels are internally encoded to integers
            0 … K-1 via LabelEncoder; the original label order is preserved
            in ``self.classes_``.

        Returns
        -------
        self : OrdinalClassifier
            Fitted instance.
        """
        # y is already integer-encoded by DataPreparer.encode_labels() upstream.
        # No secondary LabelEncoder is needed — using one would add a redundant
        # identity mapping and obscure where label ownership lives.
        y_enc            = np.asarray(y)
        self.classes_    = np.unique(y_enc)
        self.n_classes_  = len(self.classes_)
        self.classifiers_ = {}

        for k in range(self.n_classes_ - 1):
            y_binary = (y_enc > k).astype(int)
            clf      = clone(self.base_estimator)

            if self.scale_pos_weight:
                counts = Counter(y_binary)
                spw    = counts[0] / counts[1] if counts[1] > 0 else 1.0
                try:
                    clf.set_params(scale_pos_weight=spw)
                except ValueError:
                    pass

            clf.fit(X, y_binary)
            self.classifiers_[k] = clf

        return self

    def predict_proba(self, X):
        """
        Estimate class probabilities from cumulative binary predictions.

        Recovers P(y = k) for each class by differencing adjacent cumulative
        probabilities: P(y = k) = P(y > k-1) - P(y > k), with boundary
        conditions P(y = 0) = 1 - P(y > 0) and P(y = K-1) = P(y > K-2).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Row-normalised class probability estimates.
        """
        n          = X.shape[0] if hasattr(X, "shape") else len(X)
        cumprobs   = np.zeros((n, self.n_classes_ - 1))

        for k, clf in self.classifiers_.items():
            cumprobs[:, k] = clf.predict_proba(X)[:, 1]

        cumprobs = np.clip(cumprobs, 0.0, 1.0)
        for k in range(1, cumprobs.shape[1]):
            cumprobs[:, k] = np.minimum(cumprobs[:, k], cumprobs[:, k - 1])

        proba         = np.zeros((n, self.n_classes_))
        proba[:, 0]   = 1.0 - cumprobs[:, 0]
        for k in range(1, self.n_classes_ - 1):
            proba[:, k] = cumprobs[:, k - 1] - cumprobs[:, k]
        proba[:, -1]  = cumprobs[:, -1]

        row_sums = proba.sum(axis=1, keepdims=True)
        proba    = proba / np.where(row_sums == 0, 1.0, row_sums)

        return proba

    def predict(self, X):
        """
        Predict the most probable ordinal class for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels in the original label space.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def feature_importances_(self):
        """
        Mean feature importances averaged across all K-1 binary classifiers.

        Computed as the element-wise mean of ``feature_importances_`` from
        each fitted binary classifier. Only available if the base estimator
        exposes ``feature_importances_`` (e.g. tree-based models).

        Returns
        -------
        importances : np.ndarray of shape (n_features,)

        Raises
        ------
        AttributeError
            If the base estimator does not expose ``feature_importances_``.
        """
        all_importances = []
        for clf in self.classifiers_.values():
            if not hasattr(clf, "feature_importances_"):
                raise AttributeError(
                    f"{type(clf).__name__} does not expose feature_importances_. "
                    "Pass importances explicitly to _set_features() or use a "
                    "tree-based base estimator."
                )
            all_importances.append(clf.feature_importances_)

        return np.mean(all_importances, axis=0)


class OrdinalPipeline(BasePipeline):
    """
    Pipeline wrapper for ordinal classification via Frank & Hall decomposition.

    Wraps any ``BasePipeline``-compatible estimator in an ``OrdinalClassifier``
    so that the ordinal structure of the target variable (on-time < 1-30 days <
    31-60 days < 61+ days) is exploited during training. All feature selection,
    evaluation, and result-logging behaviour is inherited from ``BasePipeline``
    unchanged.

    The ``base_pipeline_cls`` parameter controls which underlying estimator is
    used. Any existing pipeline class (AdaBoostPipeline, RandomForestPipeline,
    XGBoostPipeline, etc.) can be passed; only its ``initialize_model`` logic is
    borrowed to build the inner estimator.

    Parameters
    ----------
    X_train, X_test, y_train, y_test : array-like
        Train/test splits passed through to ``BasePipeline``.
    args : any
        Forwarded to ``BasePipeline``.
    parameters : dict, optional
        Hyperparameters forwarded to the base estimator constructor.
        Also accepts ``scale_pos_weight`` (bool, default True) to control
        whether ``OrdinalClassifier`` auto-computes per-threshold class weights.
    feature_names : list of str, optional
        Column names for feature logging.
    base_estimator : sklearn-compatible estimator, optional
        A pre-instantiated unfitted estimator. When provided, ``parameters``
        is ignored for estimator construction. Useful when the caller builds
        the estimator directly (e.g. XGBClassifier with custom kwargs).

    Examples
    --------
    Using a pre-built estimator directly:

    >>> from xgboost import XGBClassifier
    >>> est = XGBClassifier(max_depth=3, learning_rate=0.01, n_estimators=500)
    >>> pipeline = OrdinalPipeline(
    ...     X_train, X_test, y_train, y_test, args,
    ...     base_estimator=est,
    ... )
    >>> pipeline.initialize_model().fit().evaluate()

    Using parameters dict (mirrors the AdaBoost / RF pattern):

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> pipeline = OrdinalPipeline(
    ...     X_train, X_test, y_train, y_test, args,
    ...     parameters={"max_depth": 10, "n_estimators": 200},
    ...     base_estimator=RandomForestClassifier(),
    ... )
    >>> pipeline.initialize_model().fit().evaluate()
    """

    def __init__(self, X_train, X_test, y_train, y_test,
                 args, parameters=None, feature_names=None,
                 base_estimator=None):
        super().__init__(X_train, X_test, y_train, y_test,
                         args, parameters, feature_names)
        self._base_estimator   = base_estimator
        self._scale_pos_weight = (parameters or {}).get("scale_pos_weight", True)

    def initialize_model(self):
        """
        Wrap the base estimator in an OrdinalClassifier and assign to self.model.

        The base estimator is sourced from ``self._base_estimator`` if provided
        at construction, otherwise it must be set before calling this method.
        ``self.model`` is set to an unfitted ``OrdinalClassifier`` instance ready
        for ``fit()``.

        Returns
        -------
        self : OrdinalPipeline

        Raises
        ------
        ValueError
            If no base estimator was supplied at construction and
            ``self._base_estimator`` is None.
        """
        if self._base_estimator is None:
            raise ValueError(
                "No base_estimator provided. Pass a pre-instantiated sklearn-compatible "
                "estimator to OrdinalPipeline(base_estimator=...) at construction time."
            )
        self.model = OrdinalClassifier(
            base_estimator   = self._base_estimator,
            scale_pos_weight = self._scale_pos_weight,
        )
        return self

    def fit(self, use_feature_selection=False, threshold="median"):
        """
        Train the ordinal model, optionally applying SelectFromModel feature selection.

        Mirrors the ``AdaBoostPipeline.fit`` pattern exactly:

        1. Fit the full ``OrdinalClassifier`` on all features.
        2. If ``use_feature_selection=True``, derive a feature mask from the
           averaged ``feature_importances_`` of the internal binary classifiers
           via ``SelectFromModel``, slice ``X_train`` / ``X_test``, capture the
           feature metadata via ``_set_features``, then retrain on the reduced set.
        3. If ``use_feature_selection=False``, call ``_set_features`` with
           method_text="none" to log that no selection was applied.

        Parameters
        ----------
        use_feature_selection : bool, default False
            Whether to apply importance-based feature selection after the first
            fit. Requires the base estimator to expose ``feature_importances_``
            (tree-based models). Will raise ``AttributeError`` for estimators
            such as KNN or MLP that do not support this.
        threshold : str or float, default "median"
            Passed to ``SelectFromModel``. Features with importance below this
            threshold are dropped. Accepts sklearn threshold strings
            (e.g. "median", "mean") or a numeric cutoff.

        Returns
        -------
        self : OrdinalPipeline

        Raises
        ------
        ValueError
            If ``initialize_model()`` has not been called prior to ``fit()``.
        AttributeError
            If ``use_feature_selection=True`` but the base estimator does not
            expose ``feature_importances_``.
        """
        if self.model is None:
            raise ValueError("Model not built. Call initialize_model() first.")

        self.model.fit(self.X_train, self.y_train)

        if use_feature_selection:
            self.selector = SelectFromModel(
                self.model, threshold=threshold, prefit=True
            )

            mask         = self.selector.get_support()
            self.X_train = self.selector.transform(self.X_train)
            self.X_test  = self.selector.transform(self.X_test)

            # Capture importances before refit overwrites them
            self._set_features(
                method_text       = "Ordinal mean feature importances",
                method_parameters = f"threshold={threshold!r}",
                mask              = mask,
            )

            self.model.fit(self.X_train, self.y_train)
        else:
            self._set_features(method_text="none")

        return self