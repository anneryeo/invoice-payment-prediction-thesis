# machine_learning/utils/io/load_results_from_folder.py
#
# Public API for loading training sessions from dated run folders.
#
# For SQLite v2 sessions the load path delegates entirely to ResultsRepository
# so that callers automatically benefit from lazy chart loading.
#
# The top-level function load_training_results() retains its original
# four-value return signature for backward compatibility; the SessionStore
# class additionally exposes a repository() accessor so Dash callbacks that
# need lazy chart hydration can skip the full DataFrame load entirely.

from __future__ import annotations

import json
import os
import pickle
import re

import pandas as pd

from .results_repository import ResultsRepository


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_DATE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}$")   # YYYY_MM_DD_##


def _has_legacy_metadata(run_folder_path: str) -> bool:
    return os.path.exists(os.path.join(run_folder_path, "metadata.json"))


def _read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC LOAD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def load_training_results(
    run_folder_path: str,
) -> tuple[pd.DataFrame, dict, dict, dict]:
    """
    Load training results, class mappings, survival results, and metadata from
    a run folder.

    Automatically detects the storage format from metadata and routes to the
    appropriate reader.  For the SQLite v2 format the returned DataFrame
    contains all columns (including chart blobs) reconstructed from the
    normalized schema — use :class:`SessionStore` and
    :meth:`ResultsRepository.load_models_dict` in new code to benefit from
    lazy chart loading.

    Parameters
    ----------
    run_folder_path : str
        Path to the dated run folder (e.g. ``Results/2025_03_05_01``).

    Returns
    -------
    model_results_df : pd.DataFrame
    class_mappings   : dict
    survival_results : dict
    metadata         : dict
    """
    db_path = os.path.join(run_folder_path, "results.db")

    # ── SQLite v2 ─────────────────────────────────────────────────────────────
    if os.path.exists(db_path):
        repo = ResultsRepository(db_path)
        return (
            repo.load_as_flat_dataframe(),
            repo.load_class_mappings(),
            repo.load_survival_results(),
            repo.load_metadata(),
        )

    # ── Legacy metadata.json path (pickle / excel) ────────────────────────────
    if not _has_legacy_metadata(run_folder_path):
        raise FileNotFoundError(
            f"No results.db or metadata.json found in {run_folder_path!r}"
        )

    metadata = _read_json(os.path.join(run_folder_path, "metadata.json"))
    fmt = metadata.get("results_format", "pickle")

    if fmt == "excel":
        model_results_df = pd.read_excel(
            os.path.join(run_folder_path, "results.xlsx")
        )
    else:
        with open(os.path.join(run_folder_path, "results.pkl"), "rb") as f:
            model_results_df = pickle.load(f)

    class_mappings = _read_json(
        os.path.join(run_folder_path, "class_mappings.json")
    )

    survival_path = os.path.join(run_folder_path, "survival_results.json")
    survival_results = (
        _read_json(survival_path) if os.path.exists(survival_path) else {}
    )

    return model_results_df, class_mappings, survival_results, metadata


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STORE
# ══════════════════════════════════════════════════════════════════════════════

class SessionStore:
    """
    Navigates and loads training sessions saved under a results root folder.

    Each session is a dated sub-folder whose name matches ``YYYY_MM_DD_##``
    (e.g. ``2025_03_05_02``).  Sessions are always ordered newest-first, so
    index ``0`` is the most recent run.

    Parameters
    ----------
    results_root : str
        Path to the directory that contains dated session folders.

    Examples
    --------
    ::

        store = SessionStore("results/")

        # Enumerate sessions
        store.list()
        # → ['2025_03_05_02', '2025_03_05_01', '2025_03_04_01']

        # Full load (backward-compatible, includes chart data)
        store.load()        # latest
        store.load(1)       # second-most-recent by index
        store.load("2025_03_04_01")  # by exact folder name

        # Lazy-loading via repository (preferred for dashboard callbacks)
        repo = store.repository()          # latest session
        models = repo.load_models_dict()   # fast: no charts loaded
        repo.hydrate_model_charts(models[key])  # on demand

        # Just the db path
        store.path()
        store.path(2)
    """

    def __init__(self, results_root: str) -> None:
        self.results_root = results_root

    # ── Discovery ─────────────────────────────────────────────────────────────

    def list(self) -> list[str]:
        """
        Return all dated session folder names, newest first.

        Returns an empty list (rather than raising) when no sessions exist.
        """
        try:
            return sorted(
                [d for d in os.listdir(self.results_root) if _DATE_RE.match(d)],
                reverse=True,
            )
        except FileNotFoundError:
            return []

    # ── Internal resolver ─────────────────────────────────────────────────────

    def _resolve(self, session: "str | int | None") -> str:
        """
        Resolve a session reference to an absolute run-folder path.

        Parameters
        ----------
        session : str, int, or None
            - ``None``  → most-recent session (index 0).
            - ``int``   → zero-based index into :meth:`list` (0 = newest).
            - ``str``   → exact folder name (e.g. ``"2025_03_05_01"``).

        Raises
        ------
        FileNotFoundError
            When no sessions exist, or the named session is not found.
        IndexError
            When an integer index is out of range.
        """
        dirs = self.list()
        if not dirs:
            raise FileNotFoundError(
                f"No dated session folders found under {self.results_root!r}"
            )

        if session is None or isinstance(session, int):
            idx = 0 if session is None else session
            if idx < 0 or idx >= len(dirs):
                raise IndexError(
                    f"Session index {idx} is out of range — "
                    f"only {len(dirs)} session(s) available."
                )
            folder = dirs[idx]
        else:
            if session not in dirs:
                raise FileNotFoundError(
                    f"Session {session!r} not found under {self.results_root!r}"
                )
            folder = session

        return os.path.join(self.results_root, folder)

    # ── Path access ───────────────────────────────────────────────────────────

    def path(self, session: "str | int | None" = None) -> str:
        """
        Return the path to ``results.db`` without loading its contents.

        Parameters
        ----------
        session : str, int, or None
            See :meth:`_resolve`.  Defaults to the most-recent session.

        Returns
        -------
        str  –  Absolute path to ``results.db``.

        Raises
        ------
        FileNotFoundError  –  When ``results.db`` is absent from the folder.
        """
        run_folder = self._resolve(session)
        db_path    = os.path.join(run_folder, "results.db")
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"results.db not found in {run_folder!r}"
            )
        return db_path

    # ── Repository access (preferred for new code) ────────────────────────────

    def repository(self, session: "str | int | None" = None) -> ResultsRepository:
        """
        Return a :class:`ResultsRepository` bound to the given session's
        database without loading any data.

        This is the preferred entry point for Dash callbacks: the leaderboard
        callback calls :meth:`ResultsRepository.load_models_dict` (fast, no
        charts), and the modal callback calls
        :meth:`ResultsRepository.hydrate_model_charts` only for the selected
        model.

        Parameters
        ----------
        session : str, int, or None
            See :meth:`_resolve`.  Defaults to the most-recent session.

        Returns
        -------
        ResultsRepository
        """
        return ResultsRepository(self.path(session))

    # ── Full load (backward-compatible) ───────────────────────────────────────

    def load(self, session: "str | int | None" = None) -> dict:
        """
        Load a training session and return its contents as a plain dict.

        For SQLite sessions this calls
        :meth:`ResultsRepository.load_as_flat_dataframe` which reconstructs
        chart columns from the normalized schema.  Prefer
        :meth:`repository` + :meth:`~ResultsRepository.load_models_dict` in
        new Dash callback code to avoid the overhead.

        Parameters
        ----------
        session : str, int, or None
            - ``None`` → most-recent session (default).
            - ``int``  → zero-based index (0 = newest).
            - ``str``  → exact folder name.

        Returns
        -------
        dict
            Keys: ``model_results_df``, ``class_mappings``,
            ``survival_results``, ``metadata``.
        """
        run_folder          = self._resolve(session)
        df, cls, surv, meta = load_training_results(run_folder)

        return {
            "model_results_df": df,
            "class_mappings":   cls,
            "survival_results": surv,
            "metadata":         meta,
        }
