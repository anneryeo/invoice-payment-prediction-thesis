# src/app/utils/app_cache.py
#
# Two-layer on-disk cache for the Invoice Drilldown screen:
#   Layer 1 — the CreditSalesProcessor feature DataFrame (credit_sales.parquet)
#   Layer 2 — per-invoice predictions from the deployed pipeline (predictions.json)
# Each layer has its own staleness check so the screen loads instantly from
# disk on repeat visits, recomputing only the layer(s) that went stale.

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.app.utils.audit_logger import log_event
from src.modules.feature_engineering.credit_sales_machine_learning import CreditSalesProcessor

_logger = logging.getLogger(__name__)

CACHE_DIR               = Path("data") / "app_cache"
CREDIT_SALES_CACHE_PATH = CACHE_DIR / "credit_sales.parquet"
CREDIT_SALES_META_PATH  = CACHE_DIR / "credit_sales_meta.json"
PREDICTIONS_CACHE_PATH  = CACHE_DIR / "predictions.json"
PREDICTIONS_META_PATH   = CACHE_DIR / "predictions_meta.json"


def _read_json_meta(path: Path, layer: str) -> dict | None:
    """Read a metadata JSON file, returning None if missing or corrupted."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        _logger.warning("[AppCache] Failed to read %s cache — falling back to recompute.", layer)
        return None


def _write_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, default=str)


# ── Layer 1 — credit sales feature DataFrame ───────────────────────────────────

def _build_credit_sales(settings: dict) -> pd.DataFrame:
    df_revenues  = pd.read_excel(settings["TrainingInput"]["REVENUES"],  engine="calamine")
    df_enrollees = pd.read_excel(settings["TrainingInput"]["ENROLLEES"], engine="calamine")

    class _Config:
        observation_end = datetime.strptime(
            settings["Training"]["observation_end"], "%Y/%m/%d"
        )

    processor = CreditSalesProcessor(
        df_revenues, df_enrollees, _Config(),
        drop_helper_columns=True,
        drop_demographic_columns=True,
        drop_plan_type_columns=False,
        drop_missing_dtp=True,
        drop_back_account_transactions=True,
        exclude_school_years=[2016, 2017, 2018],
        winsorise_dtp=True,
    )
    return processor.show_data()


def load_credit_sales(
    settings: dict,
    *,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame | None, dict]:
    """
    Returns (df, layer1_meta).
    Loads from disk if fresh, recomputes via CreditSalesProcessor if stale.
    Always recomputes when force_recompute=True.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    revenues_path  = settings["TrainingInput"]["REVENUES"]
    enrollees_path = settings["TrainingInput"]["ENROLLEES"]

    meta = None if force_recompute else _read_json_meta(CREDIT_SALES_META_PATH, "credit_sales")

    if meta is not None:
        try:
            mtime_revenues  = os.path.getmtime(revenues_path)
            mtime_enrollees = os.path.getmtime(enrollees_path)
        except OSError:
            mtime_revenues = mtime_enrollees = None

        revenues_changed  = mtime_revenues  != meta.get("source_mtime_revenues")
        enrollees_changed = mtime_enrollees != meta.get("source_mtime_enrollees")
        stale = revenues_changed or enrollees_changed or not CREDIT_SALES_CACHE_PATH.exists()

        if not stale:
            try:
                df = pd.read_parquet(CREDIT_SALES_CACHE_PATH, engine="pyarrow")
            except Exception:
                _logger.warning(
                    "[AppCache] Failed to read %s cache — falling back to recompute.", "credit_sales"
                )
            else:
                _logger.info(
                    "[AppCache] credit_sales: cache fresh (cached_at=%s, rows=%d). Loading from disk.",
                    meta.get("cached_at"), meta.get("row_count", len(df)),
                )
                return df, meta
        else:
            _logger.info(
                "[AppCache] credit_sales: stale (revenues_changed=%s, enrollees_changed=%s). Recomputing.",
                revenues_changed, enrollees_changed,
            )
    elif not force_recompute:
        _logger.info("[AppCache] credit_sales: no cache found. Computing for the first time.")

    try:
        df = _build_credit_sales(settings)
    except Exception:
        _logger.exception("Failed to build invoice cache via CreditSalesProcessor")
        return None, {}

    try:
        mtime_revenues  = os.path.getmtime(revenues_path)
        mtime_enrollees = os.path.getmtime(enrollees_path)
    except OSError:
        mtime_revenues = mtime_enrollees = None

    new_meta = {
        "cached_at":               datetime.now(timezone.utc).isoformat(),
        "source_mtime_revenues":   mtime_revenues,
        "source_mtime_enrollees":  mtime_enrollees,
        "row_count":               len(df),
    }

    try:
        df.to_parquet(CREDIT_SALES_CACHE_PATH, engine="pyarrow")
        _write_json(CREDIT_SALES_META_PATH, new_meta)
        _logger.info(
            "[AppCache] credit_sales: cache written (%d rows) → %s", len(df), CREDIT_SALES_CACHE_PATH
        )
    except Exception:
        _logger.exception("[AppCache] Failed to write %s cache.", "credit_sales")

    return df, new_meta


# ── Layer 2 — per-invoice predictions ──────────────────────────────────────────

def load_predictions(
    pipeline,
    df: pd.DataFrame,
    layer1_meta: dict,
    *,
    force_recompute: bool = False,
) -> list[dict] | None:
    """
    Returns all_rows.
    Loads from disk if fresh, re-runs inference if stale.
    Always re-runs when force_recompute=True.
    """
    if pipeline is None or df is None:
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_key = getattr(pipeline, "model_key", "unknown")

    meta = None if force_recompute else _read_json_meta(PREDICTIONS_META_PATH, "predictions")

    if meta is not None:
        credit_sales_changed = meta.get("credit_sales_cached_at") != layer1_meta.get("cached_at")
        model_changed         = meta.get("model_key") != model_key
        stale = credit_sales_changed or model_changed or not PREDICTIONS_CACHE_PATH.exists()

        if not stale:
            try:
                with open(PREDICTIONS_CACHE_PATH, "r", encoding="utf-8") as fh:
                    all_rows = json.load(fh)
            except Exception:
                _logger.warning(
                    "[AppCache] Failed to read %s cache — falling back to recompute.", "predictions"
                )
            else:
                _logger.info(
                    "[AppCache] predictions: cache fresh (cached_at=%s, rows=%d). Loading from disk.",
                    meta.get("cached_at"), meta.get("row_count", len(all_rows)),
                )
                return all_rows
        elif credit_sales_changed:
            _logger.info("[AppCache] predictions: stale (credit_sales timestamp changed). Recomputing.")
        elif model_changed:
            _logger.info(
                "[AppCache] predictions: stale (model_key changed: %s → %s). Recomputing.",
                meta.get("model_key"), model_key,
            )
    elif not force_recompute:
        _logger.info("[AppCache] predictions: no cache found. Running inference.")

    try:
        all_rows = pipeline.predict_all_invoices(df, use_cache=False)
    except Exception:
        _logger.exception("Invoice drilldown prediction failed")
        return None

    log_event("prediction_run", f"model={model_key}, n_invoices={len(df)}")

    new_meta = {
        "cached_at":              datetime.now(timezone.utc).isoformat(),
        "credit_sales_cached_at": layer1_meta.get("cached_at"),
        "model_key":              model_key,
        "row_count":              len(all_rows),
    }

    try:
        _write_json(PREDICTIONS_CACHE_PATH, all_rows)
        _write_json(PREDICTIONS_META_PATH, new_meta)
        _logger.info(
            "[AppCache] predictions: cache written (%d rows) → %s", len(all_rows), PREDICTIONS_CACHE_PATH
        )
    except Exception:
        _logger.exception("[AppCache] Failed to write %s cache.", "predictions")

    return all_rows
