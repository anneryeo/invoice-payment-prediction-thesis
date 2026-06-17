# src/app/utils/app_cache.py
#
# Three-layer on-disk cache for the Invoice Drilldown screen:
#   Layer 1 — the CreditSalesProcessor feature DataFrame (credit_sales.parquet)
#   Layer 2 — per-invoice predictions from the deployed pipeline (predictions.json)
#   Layer 3 — aggregated cash-flow month_data (cashflow.json)
# Each layer has its own staleness check so the screen loads instantly from
# disk on repeat visits, recomputing only the layer(s) that went stale.

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
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

BRACKET_DELAY = {"on_time": 0, "30_days": 30, "60_days": 60, "90_days": 91}


def _read_json_meta(path: Path, layer: str) -> dict | None:
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


# ── Layer 1 — credit sales feature DataFrame ─────────────────────────────────

class CreditSalesCache:

    def _build(self, settings: dict) -> pd.DataFrame:
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

    def load(
        self,
        settings: dict,
        *,
        force_recompute: bool = False,
    ) -> tuple[pd.DataFrame | None, dict]:
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
            df = self._build(settings)
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


# ── Layer 2 — per-invoice predictions ────────────────────────────────────────

class PredictionsCache:

    def load(
        self,
        pipeline,
        df: pd.DataFrame,
        layer1_meta: dict,
        *,
        force_recompute: bool = False,
    ) -> tuple[list[dict] | None, dict]:
        if pipeline is None or df is None:
            return None, {}

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
                    return all_rows, meta
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
            return None, {}

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

        return all_rows, new_meta


# ── Layer 3 — cash-flow aggregation ──────────────────────────────────────────

class CashFlowCache:
    CACHE_PATH = CACHE_DIR / "cashflow.json"
    META_PATH  = CACHE_DIR / "cashflow_meta.json"

    def load(
        self,
        all_rows: list[dict],
        layer2_meta: dict,
        *,
        force_recompute: bool = False,
    ) -> tuple[dict, int] | None:
        if not all_rows:
            return None

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        predictions_cached_at = layer2_meta.get("cached_at") if layer2_meta else None

        meta = None if force_recompute else _read_json_meta(self.META_PATH, "cashflow")

        if meta is not None:
            stale = (
                meta.get("predictions_cached_at") != predictions_cached_at
                or not self.CACHE_PATH.exists()
            )
            if not stale:
                try:
                    with open(self.CACHE_PATH, "r", encoding="utf-8") as fh:
                        month_data = json.load(fh)
                    skipped = meta.get("skipped", 0)
                    _logger.info("[AppCache] cashflow: cache fresh. Loading from disk.")
                    return month_data, skipped
                except Exception:
                    _logger.warning("[AppCache] Failed to read cashflow cache — recomputing.")

        _month_data: dict = defaultdict(lambda: defaultdict(float))
        skipped = 0

        for r in all_rows:
            pred_key   = r.get("_pred_key", "on_time")
            amount_raw = r.get("_amount_raw")
            due_str    = r.get("due_date", "—")
            if amount_raw is None:
                skipped += 1
                continue
            try:
                amount_val = float(amount_raw)
            except (TypeError, ValueError):
                skipped += 1
                continue
            try:
                due_dt = pd.to_datetime(due_str)
            except Exception:
                skipped += 1
                continue
            if pd.isna(due_dt):
                skipped += 1
                continue

            delay_days    = BRACKET_DELAY.get(pred_key, 0)
            expected_date = due_dt + pd.Timedelta(days=delay_days)
            month_key     = expected_date.strftime("%Y-%m")
            _month_data[month_key][pred_key] += amount_val

        month_data = {m: dict(v) for m, v in _month_data.items()}

        new_meta = {
            "cached_at":             datetime.now(timezone.utc).isoformat(),
            "predictions_cached_at": predictions_cached_at,
            "month_count":           len(month_data),
            "skipped":               skipped,
        }
        try:
            _write_json(self.CACHE_PATH, month_data)
            _write_json(self.META_PATH, new_meta)
            _logger.info(
                "[AppCache] cashflow: cache written (%d months, %d skipped) → %s",
                len(month_data), skipped, self.CACHE_PATH,
            )
        except Exception:
            _logger.exception("[AppCache] Failed to write cashflow cache.")

        return month_data, skipped


# ── Backward-compat shims ─────────────────────────────────────────────────────

_credit_sales_cache = CreditSalesCache()
_predictions_cache  = PredictionsCache()


def load_credit_sales(settings, *, force_recompute=False):
    return _credit_sales_cache.load(settings, force_recompute=force_recompute)


def load_predictions(pipeline, df, layer1_meta, *, force_recompute=False):
    return _predictions_cache.load(pipeline, df, layer1_meta, force_recompute=force_recompute)
