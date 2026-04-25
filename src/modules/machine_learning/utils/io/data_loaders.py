# machine_learning/utils/io/data_loaders.py
#
# Dashboard data-loading helpers consumed by the Dash leaderboard callbacks.
#
# The public entry point load_models_from_results(path) now delegates to
# ResultsRepository.load_models_dict() so the leaderboard loads only scalar
# metrics (experiments + metrics tables) and never touches the charts table.
#
# Chart data is fetched lazily via ResultsRepository.hydrate_model_charts()
# in the step_4 modal callback, keeping the initial dashboard render fast.

from __future__ import annotations

import json

import numpy as np

from .results_repository import ResultsRepository


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE HELPERS  (kept for any direct imports elsewhere in the codebase)
# ══════════════════════════════════════════════════════════════════════════════

def json_deserialize(value):
    """
    Transparently deserialize a value that may be a JSON-encoded dict/list
    stored as text in SQLite, or an already-native Python object.

    • None / NaN  → returned as-is
    • list / dict → returned as-is
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


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def load_models_from_results(path: str) -> dict:
    """
    Read ``results.db`` from *path* and return a ``MODELS``-compatible dict
    for the Dash leaderboard dashboard.

    This function now uses :class:`ResultsRepository` internally, which means:

    - Only the ``experiments`` and ``metrics`` tables are queried.
    - Heavy chart blobs (roc_curve, pr_curve, confusion_matrix) are **not**
      loaded; each model entry contains ``None`` placeholders for those keys.
    - Charts are fetched on demand via
      :meth:`ResultsRepository.hydrate_model_charts` in the modal callback.

    Parameters
    ----------
    path : str
        Absolute path to ``results.db``.

    Returns
    -------
    dict
        Keyed by ``"{slug}__{strategy}__{param_hash}"``.  Each value has
        ``experiment_id``, ``model``, ``balance_strategy``, ``parameters``,
        ``baseline``, and ``enhanced`` keys matching the layout expected by
        the dashboard constants and callbacks.

        Chart values under ``[phase]["evaluation"]["charts"]`` start as
        ``None`` and are hydrated lazily — call
        ``repo.hydrate_model_charts(entry)`` when the user opens the modal.
    """
    repo = ResultsRepository(path)
    models = repo.load_models_dict()

    # ── Diagnostic: log row count ─────────────────────────────────────────────
    print(f"[data_loaders] Loaded {len(models)} model entries from {path} "
          f"(charts deferred to on-demand hydration)")

    return models


def get_repository(path: str) -> ResultsRepository:
    """
    Return a :class:`ResultsRepository` bound to *path*.

    Use this in Dash callbacks that need to hydrate chart data for a specific
    model after the user opens the detail modal::

       from src.modules.machine_learning.utils.io.data_loaders import get_repository

        repo   = get_repository(db_path)
        models = load_models_from_results(db_path)  # fast, no charts
        repo.hydrate_model_charts(models[selected_key])  # on demand

    Parameters
    ----------
    path : str
        Absolute path to ``results.db``.

    Returns
    -------
    ResultsRepository
    """
    return ResultsRepository(path)
