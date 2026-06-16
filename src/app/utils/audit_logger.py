# utils/audit_logger.py
#
# Lightweight append-only audit logger.
# Writes one JSON object per line to data/app_logs/audit_log.jsonl.
# Callers import `log_event(action, details)` — no configuration needed.

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone


_DEFAULT_LOG_PATH  = os.path.join("data", "app_logs", "audit_log.jsonl")
_LEGACY_LOG_PATH   = os.path.join("data", "logs", "audit_log.jsonl")


def _migrate_legacy_log(log_path: str) -> None:
    """
    One-time copy of pre-rename audit history from data/logs to data/app_logs.

    data/logs was the log directory before it was split into app_logs (this
    file) and training_logs (model training runs). Without this, anyone with
    an existing data/logs/audit_log.jsonl would see their audit history
    silently disappear the first time log_event/read_events runs post-rename.
    """
    if log_path != _DEFAULT_LOG_PATH:
        return
    if os.path.exists(_DEFAULT_LOG_PATH) or not os.path.exists(_LEGACY_LOG_PATH):
        return
    try:
        os.makedirs(os.path.dirname(_DEFAULT_LOG_PATH) or ".", exist_ok=True)
        shutil.copy2(_LEGACY_LOG_PATH, _DEFAULT_LOG_PATH)
    except Exception as exc:
        print(f"[audit_logger] Could not migrate legacy log: {exc}")


def log_event(
    action: str,
    details: str = "",
    log_path: str = _DEFAULT_LOG_PATH,
) -> None:
    """
    Append one event record to the audit log.

    Parameters
    ----------
    action  : short action name, e.g. "prediction_run", "settings_saved"
    details : free-form description / context
    log_path: override the default log file location (mostly for tests)
    """
    try:
        _migrate_legacy_log(log_path)
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        record = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "action":    action,
            "details":   details,
        }
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        # Non-fatal — never let logging crash the app
        print(f"[audit_logger] Could not write log: {exc}")


def read_events(
    log_path: str = _DEFAULT_LOG_PATH,
    limit: int = 500,
) -> list[dict]:
    """
    Read the most-recent ``limit`` events from the log, newest-first.

    Returns an empty list if the log does not exist yet.
    """
    _migrate_legacy_log(log_path)
    if not os.path.exists(log_path):
        return []
    try:
        lines = []
        with open(log_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    lines.append(line)
        # Newest first
        lines = lines[-limit:][::-1]
        return [json.loads(line) for line in lines]
    except Exception:
        return []
