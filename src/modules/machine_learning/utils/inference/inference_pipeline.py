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

import logging
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_logger = logging.getLogger(__name__)


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
    cox_scaler : StandardScaler
        The scaler that produced the inputs ``cox_model`` was fit on
        (distinct from ``scaler`` above, which normalises the raw feature
        matrix). Required so inference can reuse the exact training-time
        transform instead of re-deriving a new scaler from whatever batch
        happens to be passed to ``predict`` — and so it never needs the
        true ``T``/``E`` survival targets, which don't exist at inference.
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
        feature_metadata: dict | None = None,
        cox_scaler: StandardScaler | None = None,
    ):
        self.scaler               = scaler
        self.cox_model            = cox_model
        self.cox_scaler           = cox_scaler
        self.time_points          = time_points
        self.classifier_pipeline  = classifier_pipeline
        self.label_encoder        = label_encoder
        self.lda_transformer      = lda_transformer
        self.model_key            = model_key
        self.features             = features
        self.parameters           = parameters or {}
        # Stores training-set statistics (e.g. plan_risk_map) so that inference
        # batches are scored using the exact distribution seen at fit time.
        self.feature_metadata     = feature_metadata or {}

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
            T=None,             # T and E are not needed: fitted_cph *and*
            E=None,             # cox_scaler are both supplied, so
            X_train=X_scaled,   # generate_survival_features never calls
            X_test=None,        # clean_survival_inputs (which requires real
            best_params=None,   # T/E) to re-derive a scaler.
            time_points=self.time_points,
            fitted_cph=self.cox_model,
            cox_scaler=self.cox_scaler,
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
            f"  model_key         = {self.model_key!r}\n"
            f"  lda_transformer   = {lda_info}\n"
            f"  time_points       = {len(self.time_points)} points\n"
            f"  classes           = {list(self.label_encoder.classes_)}\n"
            f"  feature_metadata  = {list(self.feature_metadata.keys())}\n"
            f")"
        )


# ── Module-level helpers ──────────────────────────────────────────────────────


class _LegacyArtifactUnpickler(pickle.Unpickler):
    """
    Custom unpickler that remaps legacy module paths so that old-format
    artifacts (saved before src layout was established) can be loaded
    to inspect their type without importing the full Dash application.

    ``app.screens.*`` classes are replaced with lightweight stubs so that
    pickle.load() completes; the loaded object is then inspected and, if it
    is the old flat-dict format, a descriptive ValueError is raised before
    any stub instances are used for prediction.
    """

    _stub_cache: dict = {}

    def find_class(self, module: str, name: str):
        # Remap legacy bare 'machine_learning.*' to full src path
        if module == "machine_learning" or module.startswith("machine_learning."):
            module = "src.modules." + module
            return super().find_class(module, name)

        # Stub out Dash / app layer — we only need the object shape, not its
        # callback logic, in order to detect the old-format dict.
        if module.startswith("app.") or module == "app":
            key = (module, name)
            if key not in _LegacyArtifactUnpickler._stub_cache:
                _LegacyArtifactUnpickler._stub_cache[key] = type(
                    name, (), {"__module__": module, "__reduce__": lambda s: (type(s), ())}
                )
            return _LegacyArtifactUnpickler._stub_cache[key]

        return super().find_class(module, name)


def find_deployed_model(
    model_dir: Union[str, Path],
    model_key: str | None = None,
) -> Path:
    """
    Locate a deployed InferencePipeline artifact inside *model_dir*.

    Candidates are files matching ``finalized_*.pkl`` (the survival-model
    companion ``finalized_survival_model.pkl`` is excluded because it is a
    supporting artifact, not a full inference bundle).

    Selection rules (applied in order):
    1. If *model_key* is given, return ``finalized_{model_key}.pkl``.
       Raise ``ValueError`` listing available candidates when not found.
    2. If exactly one candidate exists, return it.
    3. If multiple candidates exist, return the most recently modified file
       and emit a ``WARNING`` via the module logger naming the others.
    4. If no candidates exist, raise ``ValueError``.

    Parameters
    ----------
    model_dir : str or Path
        Directory that contains the deployed model artifacts.
    model_key : str or None
        Explicit model identifier (e.g. ``"two_stage_xgb_ada"``).
        When ``None``, the single or most-recent artifact is used.

    Returns
    -------
    Path
        Absolute path to the chosen artifact file.
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise ValueError(
            f"Model directory does not exist: {model_dir!s}. "
            "Check the 'DEPLOYED_MODELS' setting in settings.json."
        )

    candidates = sorted(
        [
            p for p in model_dir.glob("finalized_*.pkl")
            if p.name != "finalized_survival_model.pkl"
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise ValueError(
            f"No deployed model artifacts found in {model_dir!s}. "
            "Expected at least one file matching finalized_*.pkl "
            "(excluding finalized_survival_model.pkl). "
            "Run the app's Step 5 (Model Finalization) to generate one."
        )

    if model_key is not None:
        target = model_dir / f"finalized_{model_key}.pkl"
        if target in candidates:
            return target
        raise ValueError(
            f"No artifact for model_key={model_key!r} in {model_dir!s}. "
            f"Available: {[p.name for p in candidates]}"
        )

    if len(candidates) > 1:
        _logger.warning(
            "Multiple deployed artifacts found: %s. "
            "Using most recently modified: %s. "
            "Pass model_key to select explicitly.",
            [p.name for p in candidates],
            candidates[0].name,
        )

    return candidates[0]


def load_inference_pipeline(
    model_dir: Union[str, Path],
    model_key: str | None = None,
) -> InferencePipeline:
    """
    Load a deployed ``InferencePipeline`` artifact from *model_dir*.

    Only uses ``pickle.load()`` — no joblib, torch, or onnx paths. The
    artifacts in ``data/training_results/deployed_models`` are produced by
    this project's own training pipeline and are therefore trusted.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing the deployed artifacts (typically
        ``data/training_results/deployed_models``).
    model_key : str or None
        Optional model identifier.  Passed to ``find_deployed_model``.

    Returns
    -------
    InferencePipeline

    Raises
    ------
    ValueError
        * If no artifact or the named artifact is not found.
        * If the artifact is the old flat-dict format (saved before
          ``InferencePipeline`` became the deployment format).  The fix
          is to re-run the app's Step 5 (Model Finalization) to regenerate
          the artifact, then redeploy.
        * If the file cannot be unpickled (corrupt or environment mismatch).
    """
    artifact_path = find_deployed_model(model_dir, model_key=model_key)
    _logger.info("Loading InferencePipeline from %s", artifact_path)

    try:
        with open(artifact_path, "rb") as fh:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj = _LegacyArtifactUnpickler(fh).load()
    except Exception as exc:
        raise ValueError(
            f"Failed to unpickle artifact {artifact_path!s}: {exc}. "
            "The file may be corrupt, truncated, or built against an "
            "incompatible version of scikit-learn or scikit-survival."
        ) from exc

    if isinstance(obj, InferencePipeline):
        _logger.info("Loaded %r", obj)
        return obj

    # Old flat-dict format — raised as a descriptive error so the user
    # knows exactly what to do rather than seeing a generic AttributeError.
    if isinstance(obj, dict) and {"pipeline", "label_encoder"}.issubset(obj):
        raise ValueError(
            f"The artifact at {artifact_path!s} is in the pre-InferencePipeline "
            f"flat-dict format (keys: {list(obj.keys())}). It was saved before "
            "InferencePipeline became the deployment format.  To fix: open the "
            "app, go to Step 5 (Model Finalization), and re-deploy the model to "
            "regenerate a compatible artifact, then retry."
        )

    raise ValueError(
        f"Artifact {artifact_path!s} contains an unexpected object of type "
        f"{type(obj).__name__!r}.  Expected an InferencePipeline instance."
    )


def run_batch_inference(
    input_source: Union[pd.DataFrame, str, Path],
    model_dir: Union[str, Path],
    output_path: Union[str, Path, None] = None,
    batch_size: int = 1024,
    model_key: str | None = None,
    return_proba: bool = False,
) -> pd.DataFrame | None:
    """
    Run batch predictions using a deployed ``InferencePipeline``.

    Parameters
    ----------
    input_source : pd.DataFrame, str, or Path
        Raw feature data.  Accepted forms:
        * ``pd.DataFrame`` — used directly.
        * ``str`` / ``Path`` ending in ``.csv`` — loaded with
          ``pd.read_csv``.
        * ``str`` / ``Path`` ending in ``.json``, or a JSON string —
          loaded with ``pd.read_json``.
    model_dir : str or Path
        Directory containing the deployed artifact
        (typically ``data/training_results/deployed_models``).
    output_path : str, Path, or None
        When given, the results DataFrame is written to this CSV path and
        ``None`` is returned.  When ``None``, the DataFrame is returned.
    batch_size : int
        Number of rows to pass to the pipeline per chunk.  Keeps peak
        memory bounded for large inputs.
    model_key : str or None
        Optional model identifier forwarded to ``find_deployed_model``.
    return_proba : bool
        When ``True``, include per-class probability columns
        (``prob_on_time``, ``prob_30_days``, ``prob_60_days``,
        ``prob_90_days``) in addition to ``predicted_label``.

    Returns
    -------
    pd.DataFrame or None
        Columns always present:
        * ``predicted_label`` — decoded class label.
        * ``model_key`` — identifier of the artifact used.
        * ``artifact_path`` — absolute path to the artifact file.
        * ``run_timestamp`` — ISO-8601 UTC timestamp of this run.

        Additional columns when ``return_proba=True``:
        * ``prob_on_time``, ``prob_30_days``, ``prob_60_days``,
          ``prob_90_days``.

        Returns ``None`` when *output_path* is given.

    Raises
    ------
    ValueError
        * Input-source is unrecognised or cannot be parsed.
        * Input columns do not match the scaler's expected feature set:
          the error message names every missing and extra column.
        * Model artifact cannot be loaded (see ``load_inference_pipeline``).
    """
    # ── Load pipeline ─────────────────────────────────────────────────────────
    pipeline = load_inference_pipeline(model_dir, model_key=model_key)
    artifact_path = find_deployed_model(model_dir, model_key=model_key)
    run_ts = datetime.now(timezone.utc).isoformat()

    # ── Parse input ──────────────────────────────────────────────────────────
    if isinstance(input_source, pd.DataFrame):
        X_raw = input_source.copy()
    elif isinstance(input_source, (str, Path)):
        p = Path(input_source)
        if p.suffix.lower() == ".csv":
            X_raw = pd.read_csv(p)
        elif p.suffix.lower() == ".json":
            X_raw = pd.read_json(p)
        else:
            # Try as JSON string
            try:
                X_raw = pd.read_json(str(input_source))
            except Exception:
                raise ValueError(
                    f"Unsupported input_source format: {input_source!r}. "
                    "Pass a pd.DataFrame, a .csv path, or a .json path/string."
                )
    else:
        raise ValueError(
            f"input_source must be a pd.DataFrame, str, or Path — "
            f"got {type(input_source).__name__!r}."
        )

    # ── Validate columns ──────────────────────────────────────────────────────
    expected: list[str]
    if hasattr(pipeline.scaler, "feature_names_in_"):
        expected = list(pipeline.scaler.feature_names_in_)
    else:
        # Fallback: derive from X_train stored inside the classifier pipeline
        # (available for models trained before sklearn 1.0 set feature_names_in_)
        expected = list(getattr(pipeline.classifier_pipeline, "original_feature_names", []))

    if expected:
        # Keep only expected columns; validate nothing is missing
        missing = [c for c in expected if c not in X_raw.columns]
        extra   = [c for c in X_raw.columns if c not in expected]
        if missing:
            raise ValueError(
                f"Input is missing {len(missing)} column(s) required by the "
                f"deployed scaler: {missing}. "
                f"Expected columns come from pipeline.scaler.feature_names_in_."
            )
        if extra:
            _logger.warning(
                "%d extra column(s) in input will be ignored: %s", len(extra), extra
            )
        X_raw = X_raw[expected]

    n_rows = len(X_raw)
    _logger.info(
        "Running batch inference on %d rows in chunks of %d (model=%s)",
        n_rows, batch_size, pipeline.model_key,
    )

    # ── Chunked prediction ────────────────────────────────────────────────────
    label_chunks: list[np.ndarray] = []
    proba_chunks: list[pd.DataFrame] = []

    for start in range(0, n_rows, batch_size):
        chunk = X_raw.iloc[start : start + batch_size]
        label_chunks.append(pipeline.predict(chunk))
        if return_proba:
            proba_chunks.append(pipeline.predict_proba(chunk))
        _logger.info(
            "  chunk %d–%d done", start, min(start + batch_size, n_rows) - 1
        )

    # ── Assemble output ───────────────────────────────────────────────────────
    results = pd.DataFrame(index=X_raw.index)
    results["predicted_label"] = np.concatenate(label_chunks)

    if return_proba:
        proba_df = pd.concat(proba_chunks)
        for cls in proba_df.columns:
            results[f"prob_{cls}"] = proba_df[cls].values

    results["model_key"]      = pipeline.model_key
    results["artifact_path"]  = str(artifact_path.resolve())
    results["run_timestamp"]  = run_ts

    _logger.info("Batch inference complete: %d predictions.", n_rows)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out, index=True)
        _logger.info("Results written to %s", out)
        return None

    return results
