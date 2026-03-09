import ast
import hashlib
import json
import re
import sqlite3
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADER  —  reads results.db from the latest dated folder under Results/
# ══════════════════════════════════════════════════════════════════════════════

# ── Helpers ───────────────────────────────────────────────────────────────────

def json_deserialize(value):
    """
    Transparently deserialize a value that may be a JSON-encoded dict/list
    stored as text in SQLite, or an already-native Python object.

    • None / NaN  → returned as-is
    • list / dict → returned as-is  (already decoded by sqlite3 row_factory)
    • str         → attempted JSON parse; original string returned on failure
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _normalise_params(raw) -> dict:
    """Convert any params representation to a plain {str: scalar} dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (list, tuple)):
        # list/tuple of 2-tuples: [('key', val), ...]
        try:
            result = {}
            for item in raw:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    result[str(item[0])] = item[1]
            if result:
                return result
        except Exception:
            pass
    if isinstance(raw, str) and raw.strip():
        # Try single-quoted dict → JSON
        try:
            cleaned = (raw.strip()
                       .replace("'", '"')
                       .replace("True", "true")
                       .replace("False", "false")
                       .replace("None", "null"))
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        # Try Python literal_eval for list-of-tuples repr
        try:
            evaled = ast.literal_eval(raw.strip())
            return _normalise_params(evaled)
        except Exception:
            pass
    return {}


def load_raw_rows(path: str) -> list[dict]:
    """
    Load result rows from a SQLite database at *path*.
    Returns a list of plain dicts keyed by column name, with JSON-encoded
    columns transparently deserialized.

    The database must contain a table named ``results`` whose columns match
    the schema written by the training pipeline.  Dict/list columns are stored
    as JSON text and are decoded here before the caller sees them.
    """
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row          # column-name access
    try:
        cur = con.execute("SELECT * FROM results")
        columns = [desc[0] for desc in cur.description]
        raw_rows = []
        for sqlite_row in cur.fetchall():
            row = {}
            for col in columns:
                v = sqlite_row[col]
                row[col] = json_deserialize(v)
            raw_rows.append(row)
    finally:
        con.close()

    return raw_rows


def load_models_from_results(path: str) -> dict:
    """
    Read results.db from the given path and return a MODELS-compatible dict.
    Dict/list columns (confusion_matrix, roc_curve, pr_curve, feature_selected)
    are JSON-deserialized transparently by load_raw_rows.
    """
    raw_rows = load_raw_rows(path)

    # ── Diagnostic: log curve lengths ────────────────────────────────────────
    if raw_rows:
        sample = raw_rows[0]
        for col in ("baseline_roc_curve", "baseline_pr_curve",
                    "enhanced_roc_curve", "enhanced_pr_curve"):
            v = sample.get(col, "")
            print(f"[step4] {col} (sqlite) length: {len(str(v)) if v else 0}")

    # ── Build MODELS dict ─────────────────────────────────────────────────────
    # Each row is uniquely identified by (model, balance_strategy, parameters).
    # We include a short MD5 hash of the serialized parameters in the key so
    # that rows sharing the same model+strategy but differing only in
    # hyperparameters do not silently overwrite each other.
    models: dict = {}
    for row in raw_rows:

        def _get(col):
            v = row.get(col)
            return v if v is not None else ""

        model_name: str = str(_get("model"))
        balance: str    = str(_get("balance_strategy")) if _get("balance_strategy") else "none"

        params: dict = _normalise_params(json_deserialize(_get("parameters")))
        params_sig   = json.dumps(params, sort_keys=True) if params else "default"
        param_hash   = hashlib.md5(params_sig.encode()).hexdigest()[:6]

        slug = re.sub(r"\s+", "_", model_name).lower()
        key  = f"{slug}__{balance}__{param_hash}"

        def _section(prefix: str) -> dict:
            def _float(col):
                v = _get(f"{prefix}_{col}")
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return 0.0

            metrics = {
                "accuracy":        _float("accuracy"),
                "precision_macro": _float("precision_macro"),
                "recall_macro":    _float("recall_macro"),
                "f1_macro":        _float("f1_macro"),
                "roc_auc_macro":   _float("roc_auc_macro"),
            }
            # roc_curve and pr_curve are already dicts after JSON deserialization
            charts = {
                "confusion_matrix": json_deserialize(_get(f"{prefix}_confusion_matrix")),
                "roc_curve":        _get(f"{prefix}_roc_curve"),   # kept raw for _parse_curve
                "pr_curve":         _get(f"{prefix}_pr_curve"),    # kept raw for _parse_curve
            }
            raw_selected = json_deserialize(_get(f"{prefix}_feature_selected"))
            features: list = raw_selected if isinstance(raw_selected, list) else []
            feature_method: str = str(_get(f"{prefix}_feature_method") or "").strip()

            return {
                "evaluation": {"metrics": metrics, "charts": charts},
                "features": features,
                "feature_method": feature_method,
            }

        models[key] = {
            "model":            model_name,
            "balance_strategy": balance,
            "parameters":       params,
            "baseline":         _section("baseline"),
            "enhanced":         _section("enhanced"),
        }

    return models