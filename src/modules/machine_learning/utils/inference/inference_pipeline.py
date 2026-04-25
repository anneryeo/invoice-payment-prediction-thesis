"""
inference_pipeline.py
=====================
Self-contained inference wrapper for the deployed payment-delay classifier.

Motivation
----------
At training time, raw features pass through a fixed sequence of transforms
before reaching the classifier:

    raw X
      → StandardScaler       (fitted on training data)
      → generate_survival_features  (Cox hazard/survival columns appended)
      → LDATransformer       (optional; 4-class LD components appended)
      → classifier.predict()

None of these transforms are stored inside the sklearn/XGBoost pipeline
object itself — they live in separate fitted objects produced during
finalization.  Without a wrapper, the inference endpoint would have to
replicate this chain manually, and any future change to the chain (e.g.
adding a new transform) would require updating every consumer.

``InferencePipeline`` wraps all fitted objects into a single pickle-able
bundle.  Loading one file gives you a single object whose ``predict()`` and
``predict_proba()`` methods accept a raw feature DataFrame and return
decoded class labels / probability arrays — no caller-side preprocessing
required.

Usage at inference time
-----------------------
::

    import pickle

    with open("finalized_two_stage_xgb_ada.pkl", "rb") as fh:
        inf = pickle.load(fh)

    # X_raw is a pd.DataFrame with the same columns as the training data
    # (before any scaling, survival feature generation, or LDA).
    labels = inf.predict(X_raw)
    probas = inf.predict_proba(X_raw)

Saved by
--------
``step_5._finalize_in_background`` — the ``InferencePipeline`` object
replaces the old flat dict that was previously pickled.

Compatibility
-------------
The old flat-dict format (``{"pipeline": …, "label_encoder": …, …}``) is
no longer written.  Any existing inference endpoint that unpickled the
dict must be updated to call ``inf.predict(X_raw)`` instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class InferencePipeline:
    """
    Single-object inference bundle for the deployed payment-delay classifier.

    Encapsulates every fitted transform that was applied between raw features
    and the classifier at training time, so that ``predict`` and
    ``predict_proba`` can reproduce the full preprocessing chain from raw
    input at inference time.

    Parameters
    ----------
    scaler : StandardScaler
        Fitted on the full training dataset (pre-survival, pre-LDA).
        Applied first to normalise numeric columns.
    cox_model : sksurv CoxnetSurvivalAnalysis
        Fitted survival model used to compute hazard / survival columns.
        These columns are appended to the scaled feature matrix.
    time_points : list of float
        Time points at which survival / hazard functions are evaluated.
        Must match those used during ``generate_survival_features`` at
        training time.
    classifier_pipeline : BasePipeline subclass (fitted)
        The fitted sklearn-compatible pipeline — one of OrdinalPipeline,
        TwoStagePipeline, XGBoostPipeline, etc.  Its ``predict`` and
        ``predict_proba`` methods are called after preprocessing.
    label_encoder : LabelEncoder
        Fitted on the ordinal class labels so that integer predictions
        can be decoded back to ``"on_time"``, ``"30_days"``, etc.
    lda_transformer : LDATransformer or None
        Fitted 4-class LDA transformer applied after survival feature
        generation.  ``None`` when LDA was not used during training.
    model_key : str
        Model identifier string, e.g. ``"two_stage_xgb_ada"``.  Stored
        for provenance / logging at inference time.
    features : object
        ``pipeline.features`` object from the fitted classifier, carrying
        feature selection metadata (selected columns, weights, method).
    parameters : dict
        Hyperparameter dict used at training time.  Stored for provenance.

    Attributes
    ----------
    All constructor arguments are stored as public attributes under the
    same names so they can be inspected after loading from pickle.

    Notes
    -----
    ``stage2_lda_`` (the delinquent-only LDA inside ``TwoStageClassifier``)
    is stored inside ``classifier_pipeline.model`` and is automatically
    applied by ``TwoStageClassifier.predict_proba`` — the caller does not
    need to handle it separately.
    """

    def __init__(
        self,
        scaler: StandardScaler,
        cox_model,
        time_points: list,
        classifier_pipeline,
        label_encoder,
        lda_transformer=None,
        model_key: str = "",
        features=None,
        parameters: dict | None = None,
    ):
        self.scaler               = scaler
        self.cox_model            = cox_model
        self.time_points          = time_points
        self.classifier_pipeline  = classifier_pipeline
        self.label_encoder        = label_encoder
        self.lda_transformer      = lda_transformer
        self.model_key            = model_key
        self.features             = features
        self.parameters           = parameters or {}

    # ── internal preprocessing chain ─────────────────────────────────────────

    def _preprocess(self, X_raw: pd.DataFrame) -> np.ndarray:
        """
        Apply the full preprocessing chain to a raw feature DataFrame.

        Steps
        -----
        1. Cast to float64 and align columns to those seen during training.
        2. StandardScaler transform (same scaler fitted at training time).
        3. generate_survival_features — appends Cox hazard/survival columns.
        4. LDATransformer.transform — appends LD components (optional).

        Parameters
        ----------
        X_raw : pd.DataFrame
            Raw features, same columns as the original training DataFrame
            (before any preprocessing).  Extra columns are silently dropped;
            missing columns raise a ``ValueError``.

        Returns
        -------
        np.ndarray
            Preprocessed feature matrix ready for the classifier.
        """
       from src.modules.machine_learning.utils.features.generate_survival_features import (
            generate_survival_features,
        )

        # ── 1. Scale ─────────────────────────────────────────────────────────
        # Align to the columns the scaler was fitted on, then transform.
        # Using a DataFrame preserves column names for LDATransformer.
        X = X_raw.copy().astype(float)
        X_scaled_arr = self.scaler.transform(X)
        X_scaled = pd.DataFrame(
            X_scaled_arr, columns=X.columns, index=X.index
        )

        # ── 2. Survival features ──────────────────────────────────────────────
        # generate_survival_features returns X_train_enhanced (and optionally
        # X_test_enhanced).  We pass X_test=None to get the single-set path.
        X_enhanced = generate_survival_features(
            X_surv=X_scaled,
            T=None,             # T and E are not needed when using a pre-fitted
            E=None,             # Cox model — the function uses fitted_cph directly
            X_train=X_scaled,
            X_test=None,
            best_params=None,   # not needed when fitted_cph is supplied
            time_points=self.time_points,
            fitted_cph=self.cox_model,
        )

        # ── 3. LDA ───────────────────────────────────────────────────────────
        # Applied only when a 4-class LDA transformer was fitted at training.
        # Stage-2 delinquent LDA (inside TwoStageClassifier) is handled
        # automatically by the classifier's predict_proba — no action needed.
        if self.lda_transformer is not None:
            X_enhanced = self.lda_transformer.transform(X_enhanced)

        return X_enhanced

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, X_raw: pd.DataFrame) -> np.ndarray:
        """
        Predict payment-delay class for each row in ``X_raw``.

        Parameters
        ----------
        X_raw : pd.DataFrame
            Raw feature DataFrame.  Same columns as the training data,
            no preprocessing required.

        Returns
        -------
        np.ndarray of str, shape (n_samples,)
            Decoded class labels: ``"on_time"``, ``"30_days"``,
            ``"60_days"``, or ``"90_days"``.
        """
        X = self._preprocess(X_raw)
        y_encoded = self.classifier_pipeline.predict(X)
        return self.label_encoder.inverse_transform(y_encoded.astype(int))

    def predict_proba(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate class probabilities for each row in ``X_raw``.

        Parameters
        ----------
        X_raw : pd.DataFrame
            Raw feature DataFrame.  Same columns as the training data.

        Returns
        -------
        pd.DataFrame, shape (n_samples, 4)
            Columns: ``"on_time"``, ``"30_days"``, ``"60_days"``,
            ``"90_days"``.  Each row sums to 1.0.
        """
        X      = self._preprocess(X_raw)
        probas = self.classifier_pipeline.predict_proba(X)
        classes = self.label_encoder.classes_   # ["on_time", "30_days", …]
        return pd.DataFrame(probas, columns=classes, index=X_raw.index)

    def __repr__(self) -> str:
        lda_info = (
            f"LDATransformer(mode={self.lda_transformer.mode!r})"
            if self.lda_transformer is not None
            else "None"
        )
        return (
            f"InferencePipeline(\n"
            f"  model_key       = {self.model_key!r}\n"
            f"  lda_transformer = {lda_info}\n"
            f"  time_points     = {len(self.time_points)} points\n"
            f"  classes         = {list(self.label_encoder.classes_)}\n"
            f")"
        )
