"""
Smoke tests for run_batch_inference and related helpers.

Covers:
  - load / find with a synthetic InferencePipeline artifact
  - correct output shape and dtypes
  - error for missing model directory
  - error for non-InferencePipeline (invalid) artifact
  - error for input with missing / extra columns
  - error for old flat-dict format artifact
"""

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Make sure project root is on sys.path when running from anywhere
import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.modules.machine_learning.utils.inference.inference_pipeline import (
    InferencePipeline,
    find_deployed_model,
    load_inference_pipeline,
    run_batch_inference,
)


# ── Synthetic helpers ─────────────────────────────────────────────────────────


_FEATURES = ["feat_a", "feat_b", "feat_c"]
_CLASSES  = ["on_time", "30_days", "60_days", "90_days"]
_N        = 20


class _TinyCoxModel:
    """Minimal Cox model stub — returns constant hazard/survival values."""

    def predict_cumulative_hazard_function(self, X):
        from sklearn.utils import check_array
        n = len(X)

        class _StepFn:
            def __init__(self, x_vals, y_vals):
                self.x = np.array(x_vals)
                self.y = np.array(y_vals)

            def __call__(self, t):
                return np.interp(t, self.x, self.y)

        return [_StepFn([0, 100], [0.0, 0.5]) for _ in range(n)]

    def predict_survival_function(self, X):
        n = len(X)

        class _StepFn:
            def __init__(self, x_vals, y_vals):
                self.x = np.array(x_vals)
                self.y = np.array(y_vals)

            def __call__(self, t):
                return np.interp(t, self.x, self.y)

        return [_StepFn([0, 100], [1.0, 0.5]) for _ in range(n)]


class _TinyClassifier:
    """Minimal classifier stub — always predicts class 0."""

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        n = len(X)
        p = np.zeros((n, 4))
        p[:, 0] = 1.0
        return p


class _SyntheticInferencePipeline(InferencePipeline):
    """
    Subclass that skips the sksurv survival-feature step so the smoke tests
    do not require scikit-survival to be installed in every test environment.

    Defined at module level so pickle can find the class by qualified name.
    """

    def _preprocess(self, X_raw: pd.DataFrame) -> np.ndarray:
        X = X_raw.copy().astype(float)
        return self.scaler.transform(X)


def _make_synthetic_pipeline(tmp_dir: Path) -> Path:
    """
    Fit a minimal InferencePipeline on synthetic data and pickle it.

    Returns the path to the written artifact.
    """
    rng = np.random.default_rng(42)
    X_raw = pd.DataFrame(rng.normal(size=(_N, 3)), columns=_FEATURES)

    scaler = StandardScaler()
    scaler.fit(X_raw)

    le = LabelEncoder()
    le.fit(_CLASSES)

    pipeline = _SyntheticInferencePipeline(
        scaler=scaler,
        cox_model=_TinyCoxModel(),
        time_points=[10, 30, 60],
        classifier_pipeline=_TinyClassifier(),
        label_encoder=le,
        lda_transformer=None,
        model_key="synthetic_test",
        features=None,
        parameters={},
        feature_metadata={},
    )

    artifact_path = tmp_dir / "finalized_synthetic_test.pkl"
    with open(artifact_path, "wb") as fh:
        pickle.dump(pipeline, fh)

    return artifact_path


def _make_synthetic_df(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(n, 3)), columns=_FEATURES)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestFindDeployedModel:
    def test_finds_single_artifact(self, tmp_path):
        _make_synthetic_pipeline(tmp_path)
        found = find_deployed_model(tmp_path)
        assert found.exists()
        assert found.name == "finalized_synthetic_test.pkl"

    def test_model_key_match(self, tmp_path):
        _make_synthetic_pipeline(tmp_path)
        found = find_deployed_model(tmp_path, model_key="synthetic_test")
        assert found.name == "finalized_synthetic_test.pkl"

    def test_model_key_not_found_raises(self, tmp_path):
        _make_synthetic_pipeline(tmp_path)
        with pytest.raises(ValueError, match="No artifact for model_key"):
            find_deployed_model(tmp_path, model_key="nonexistent")

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            find_deployed_model(tmp_path / "no_such_dir")

    def test_no_artifacts_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No deployed model artifacts found"):
            find_deployed_model(tmp_path)

    def test_survival_model_excluded(self, tmp_path):
        # Survival pkl alone should NOT be returned as a candidate
        (tmp_path / "finalized_survival_model.pkl").write_bytes(b"x")
        with pytest.raises(ValueError, match="No deployed model artifacts found"):
            find_deployed_model(tmp_path)

    def test_multiple_artifacts_picks_newest(self, tmp_path):
        p1 = tmp_path / "finalized_model_a.pkl"
        p2 = tmp_path / "finalized_model_b.pkl"
        p1.write_bytes(b"a")
        p2.write_bytes(b"b")
        import time; time.sleep(0.05)
        p2.touch()  # ensure p2 is newer
        found = find_deployed_model(tmp_path)
        assert found.name == "finalized_model_b.pkl"


class TestLoadInferencePipeline:
    def test_loads_valid_artifact(self, tmp_path):
        _make_synthetic_pipeline(tmp_path)
        pipeline = load_inference_pipeline(tmp_path)
        assert isinstance(pipeline, InferencePipeline)
        assert pipeline.model_key == "synthetic_test"

    def test_invalid_artifact_raises(self, tmp_path):
        bad = tmp_path / "finalized_bad.pkl"
        with open(bad, "wb") as fh:
            pickle.dump({"not": "an inference pipeline"}, fh)
        with pytest.raises(ValueError, match="unexpected object"):
            load_inference_pipeline(tmp_path)

    def test_old_flat_dict_raises_descriptive_error(self, tmp_path):
        old_fmt = tmp_path / "finalized_legacy.pkl"
        from sklearn.preprocessing import LabelEncoder as LE
        with open(old_fmt, "wb") as fh:
            pickle.dump({"pipeline": object(), "label_encoder": LE()}, fh)
        with pytest.raises(ValueError, match="pre-InferencePipeline"):
            load_inference_pipeline(tmp_path)

    def test_corrupt_artifact_raises(self, tmp_path):
        bad = tmp_path / "finalized_corrupt.pkl"
        bad.write_bytes(b"\x80\x05garbage bytes that are not valid pickle")
        with pytest.raises(ValueError, match="Failed to unpickle"):
            load_inference_pipeline(tmp_path)


class TestRunBatchInference:
    def test_output_shape(self, tmp_path):
        _make_synthetic_pipeline(tmp_path)
        X = _make_synthetic_df(15)
        result = run_batch_inference(X, model_dir=tmp_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 15
        assert "predicted_label" in result.columns
        assert "model_key" in result.columns
        assert "artifact_path" in result.columns
        assert "run_timestamp" in result.columns

    def test_return_proba_columns(self, tmp_path):
        _make_synthetic_pipeline(tmp_path)
        X = _make_synthetic_df(5)
        result = run_batch_inference(X, model_dir=tmp_path, return_proba=True)
        for cls in ["on_time", "30_days", "60_days", "90_days"]:
            assert f"prob_{cls}" in result.columns

    def test_chunked_matches_full(self, tmp_path):
        _make_synthetic_pipeline(tmp_path)
        X = _make_synthetic_df(30)
        r_full  = run_batch_inference(X, model_dir=tmp_path, batch_size=100)
        r_chunk = run_batch_inference(X, model_dir=tmp_path, batch_size=7)
        pd.testing.assert_series_equal(
            r_full["predicted_label"].reset_index(drop=True),
            r_chunk["predicted_label"].reset_index(drop=True),
        )

    def test_csv_input(self, tmp_path):
        _make_synthetic_pipeline(tmp_path)
        X = _make_synthetic_df(8)
        csv_path = tmp_path / "input.csv"
        X.to_csv(csv_path, index=False)
        result = run_batch_inference(csv_path, model_dir=tmp_path)
        assert len(result) == 8

    def test_output_path_writes_csv(self, tmp_path):
        _make_synthetic_pipeline(tmp_path)
        X = _make_synthetic_df(5)
        out = tmp_path / "output" / "preds.csv"
        ret = run_batch_inference(X, model_dir=tmp_path, output_path=out)
        assert ret is None
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == 5

    def test_missing_model_dir_raises(self, tmp_path):
        X = _make_synthetic_df(3)
        with pytest.raises(ValueError, match="does not exist"):
            run_batch_inference(X, model_dir=tmp_path / "no_such_dir")

    def test_missing_columns_raises(self, tmp_path):
        _make_synthetic_pipeline(tmp_path)
        X_bad = _make_synthetic_df(5).rename(columns={"feat_a": "wrong_col"})
        with pytest.raises(ValueError, match="missing.*column"):
            run_batch_inference(X_bad, model_dir=tmp_path)

    def test_extra_columns_ignored(self, tmp_path):
        _make_synthetic_pipeline(tmp_path)
        X = _make_synthetic_df(5)
        X["extra_col"] = 0.0
        result = run_batch_inference(X, model_dir=tmp_path)
        assert len(result) == 5
