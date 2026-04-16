# app/screens/comparative_model_dashboard_template/utils/session_loader.py
#
# Single source of truth for loading a training session into the dashboard's
# global state (MODELS, CLASS_LABELS, _ACTIVE_REPO).
#
# Why this file exists
# ────────────────────
# Previously screen_1.py and screen_2.py each independently reproduced the
# same four-step activation sequence, and neither called set_active_repo() —
# meaning chart hydration was permanently broken. Centralising the sequence
# here means:
#
#   • set_active_repo() can never be forgotten again — it's part of the
#     definition of "activate a session".
#   • _store is instantiated once instead of twice.
#   • class_mappings are read via repo.load_class_mappings() (a single-row
#     blob query) instead of repo.load_as_flat_dataframe() which was
#     deserializing every chart blob for every experiment just to obtain
#     the class label mapping.

from __future__ import annotations

from machine_learning.utils.io.load_results_from_folder import SessionStore
from utils.data_loaders.read_settings_json import read_settings_json
from app.screens.comparative_model_dashboard_template.constants import (
    MODELS,
    set_class_labels,
    set_active_repo,
)


# ── Shared store (one instance for the whole app) ─────────────────────────────

_store = SessionStore(read_settings_json()["Training"]["RESULTS_ROOT"])


def get_store() -> SessionStore:
    """
    Return the shared SessionStore.

    Exposed so that screen_2.py can call ``get_store().list()`` for the
    session dropdown without instantiating its own store.
    """
    return _store


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def activate_session(
    session: "str | int | None" = None,
    *,
    clear_first: bool = False,
) -> int:
    """
    Load a training session into the dashboard's global state.

    Performs the complete activation sequence in the correct order:

    1. Resolve the session folder and open a ``ResultsRepository``.
    2. Load the lightweight models dict (metrics only, no chart blobs).
    3. Read class mappings from the tiny blob table.
    4. Optionally clear MODELS before updating (use ``clear_first=True``
       when switching sessions so stale entries from the previous session
       are removed).
    5. Write models into the global ``MODELS`` dict.
    6. Populate ``CLASS_LABELS`` via ``set_class_labels()``.
    7. Register the repo as the active repo via ``set_active_repo()`` so
       that ``update_charts`` in core.py can hydrate chart data on demand.

    Step 7 is the critical one that was missing before this refactor.
    Without it ``get_active_repo()`` always returns ``None`` and every
    chart render receives ``None`` placeholder data.

    Parameters
    ----------
    session : str, int, or None
        Passed directly to ``SessionStore._resolve()``:
        - ``None``  → most-recent session (default).
        - ``int``   → zero-based index (0 = newest).
        - ``str``   → exact folder name, e.g. ``"2025_03_05_01"``.
    clear_first : bool, default False
        When ``True``, ``MODELS.clear()`` is called before the new
        entries are inserted.  Pass ``True`` when switching sessions on
        the analysis screen so stale model keys from the previous session
        are gone.  Pass ``False`` (the default) for the initial Step 4
        load where MODELS starts empty.

    Returns
    -------
    int
        Number of model entries loaded into MODELS.

    Raises
    ------
    FileNotFoundError
        When no session folders exist or the specified session is not found.

    Examples
    --------
    Initial load (Step 4 setup flow)::

        n = activate_session()            # most-recent session
        print(f"Loaded {n} models")

    Session switch (analysis screen)::

        n = activate_session("2025_03_05_01", clear_first=True)
    """
    repo           = _store.repository(session)
    models         = repo.load_models_dict()
    class_mappings = repo.load_class_mappings()

    if clear_first:
        MODELS.clear()

    MODELS.update(models)
    set_class_labels(class_mappings)
    set_active_repo(repo)          # ← the wire that was always missing

    return len(MODELS)
