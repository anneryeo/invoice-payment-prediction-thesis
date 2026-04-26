import os
import glob
import pickle

from src.utils.data_loaders.read_settings_json import read_settings_json


def _get_deployed_models_dir() -> str:
    return read_settings_json()["Training"]["DEPLOYED_MODELS"]


def _find_classification_model_path(deployed_dir: str) -> str | None:
    """
    Return the path of the finalized classification model pickle, or None.
    The file is named finalized_<model_key>.pkl — anything that is NOT
    the survival model.
    """
    pattern    = os.path.join(deployed_dir, "finalized_*.pkl")
    candidates = [
        p for p in glob.glob(pattern)
        if not p.endswith("finalized_survival_model.pkl")
    ]
    return candidates[0] if candidates else None


def _get_survival_model_path(deployed_dir: str) -> str:
    return os.path.join(deployed_dir, "finalized_survival_model.pkl")


# ── Public API ────────────────────────────────────────────────────────────────

def has_trained_models() -> bool:
    """
    Return True if both the classification model and the survival model
    pickle files exist in the DEPLOYED_MODELS directory.
    """
    deployed_dir  = _get_deployed_models_dir()
    survival_path = _get_survival_model_path(deployed_dir)
    classify_path = _find_classification_model_path(deployed_dir)

    return (
        classify_path is not None and os.path.exists(classify_path)
        and os.path.exists(survival_path)
    )


def validate_trained_models() -> dict:
    """
    Check that both model files exist AND can be loaded without error.

    Returns a dict:
        {
            "ok": bool,
            "classification": {
                "path":   str | None,
                "exists": bool,
                "loaded": bool,
                "error":  str | None,
            },
            "survival": {
                "path":   str,
                "exists": bool,
                "loaded": bool,
                "error":  str | None,
            },
        }
    """
    deployed_dir  = _get_deployed_models_dir()
    survival_path = _get_survival_model_path(deployed_dir)
    classify_path = _find_classification_model_path(deployed_dir)

    def _try_load(path: str | None) -> tuple[bool, str | None]:
        if path is None or not os.path.exists(path):
            return False, "File not found"
        try:
            with open(path, "rb") as fh:
                pickle.load(fh)
            return True, None
        except Exception as exc:
            return False, str(exc)

    classify_loaded, classify_err = _try_load(classify_path)
    survival_loaded, survival_err = _try_load(survival_path)

    return {
        "ok": classify_loaded and survival_loaded,
        "classification": {
            "path":   classify_path,
            "exists": classify_path is not None and os.path.exists(classify_path),
            "loaded": classify_loaded,
            "error":  classify_err,
        },
        "survival": {
            "path":   survival_path,
            "exists": os.path.exists(survival_path),
            "loaded": survival_loaded,
            "error":  survival_err,
        },
    }