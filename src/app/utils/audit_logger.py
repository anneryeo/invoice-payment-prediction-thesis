# utils/audit_logger.py
#
# Lightweight append-only audit logger.
# Writes one JSON object per line to data/logs/audit_log.jsonl.
# Callers import `log_event(action, details)` — no configuration needed.

from __future__ import annotations

import json
import os
from datetime import datetime, timezone


_DEFAULT_LOG_PATH = os.path.join("data", "logs", "audit_log.jsonl")


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
