# machine_learning/utils/io/analysis/analyzer.py
#
# ResultsAnalyzer — fluent API entry point for ML experiment analysis.
#
# Composes SessionStore (filesystem navigation) and ResultsRepository
# (SQLite I/O) without inheriting from either.  Both composed objects
# are publicly accessible as escape hatches for callers that need
# lower-level access (e.g. Dash callbacks).

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ..load_results_from_folder import SessionStore
from ..results_repository import ResultsRepository
from .quality import QualityReport, run_quality_checks
from .registry import (
    FAMILY_MAP,
    MODEL_DISPLAY,
    strategy_label as _build_strategy_label,
)
from .result_set import ComparisonResult, ResultSet


class ResultsAnalyzer:
    """
    Fluent API for exploring ML experiment results stored in SQLite.

    Composes :class:`SessionStore` for session discovery and
    :class:`ResultsRepository` for database access.  Both are publicly
    accessible for callers that need the lower-level objects::

        ra = ResultsAnalyzer("data/training_results/")
        ra.store       # → SessionStore (filesystem navigation)
        ra.repo        # → ResultsRepository (current session DB)

    Parameters
    ----------
    results_root : str
        Path to the directory containing dated session folders.
    session : str | int | None
        Which session to bind to initially.
        - ``None`` → most recent (default)
        - ``int``  → zero-based index (0 = newest)
        - ``str``  → exact folder name (e.g. ``"2026_04_25_01"``)

    Examples
    --------
    ::

        ra = ResultsAnalyzer("data/training_results/")

        # Quick leaderboard
        ra.top(10)

        # Best per family with confusion matrices
        ra.best_per("family").top(3).charts("confusion_matrix").plot()

        # Statistical comparison
        ts = ra.family("Two-Stage").top(1)
        base = ra.family("Base").top(1)
        print(ts.compare(base, test="mannwhitney"))
    """

    def __init__(
        self,
        results_root: str,
        session: Union[str, int, None] = None,
    ) -> None:
        # ── Composed objects (public) ─────────────────────────────────────────
        self.store = SessionStore(results_root)

        # ── Session binding ───────────────────────────────────────────────────
        self._session = session
        self._repo: Optional[ResultsRepository] = None
        self._df: Optional[pd.DataFrame] = None
        self._metadata: Optional[dict] = None
        self._class_mappings: Optional[dict] = None
        self._survival: Optional[dict] = None

    # ── Alternative constructors ──────────────────────────────────────────────

    @classmethod
    def from_repository(cls, repo: ResultsRepository) -> ResultsAnalyzer:
        """
        Create an analyzer directly from a ResultsRepository, bypassing
        SessionStore.  Useful for testing or when you already have a DB path.
        """
        # Create instance without a valid results_root
        instance = cls.__new__(cls)
        instance.store = None  # type: ignore[assignment]
        instance._session = None
        instance._repo = repo
        instance._df = None
        instance._metadata = None
        instance._class_mappings = None
        instance._survival = None
        return instance

    @classmethod
    def from_db(cls, db_path: str) -> ResultsAnalyzer:
        """
        Create an analyzer bound to a specific ``results.db`` file.

        ::

            ra = ResultsAnalyzer.from_db("data/training_results/2026_04_25_01/results.db")
        """
        return cls.from_repository(ResultsRepository(db_path))

    # ── Session management ────────────────────────────────────────────────────

    @property
    def sessions(self) -> list[str]:
        """List all available session folder names, newest first."""
        if self.store is None:
            return []
        return self.store.list()

    @property
    def current_session(self) -> Optional[str]:
        """Name of the currently bound session folder."""
        if self.store is None:
            return None
        dirs = self.store.list()
        if not dirs:
            return None
        if self._session is None or isinstance(self._session, int):
            idx = 0 if self._session is None else self._session
            return dirs[idx] if idx < len(dirs) else None
        return self._session if self._session in dirs else None

    def use_session(self, session: Union[str, int, None] = None) -> ResultsAnalyzer:
        """
        Switch to a different session.  Clears all cached data.

        Parameters
        ----------
        session : str | int | None
            See :class:`ResultsAnalyzer` constructor.

        Returns
        -------
        self — for chaining: ``ra.use_session(1).top(5)``
        """
        self._session = session
        self._repo = None
        self._df = None
        self._metadata = None
        self._class_mappings = None
        self._survival = None
        return self

    # ── Lazy-loaded properties ────────────────────────────────────────────────

    @property
    def repo(self) -> ResultsRepository:
        """The :class:`ResultsRepository` for the current session."""
        if self._repo is None:
            if self.store is None:
                raise RuntimeError("No store configured — use from_db() or from_repository()")
            self._repo = self.store.repository(self._session)
        return self._repo

    @property
    def df(self) -> pd.DataFrame:
        """
        Full experiments + metrics DataFrame with derived columns:
        ``strategy_label``, ``model_display``, ``family``, ``f1_lift``.
        """
        if self._df is None:
            self._df = self._build_df()
        return self._df.copy()

    @property
    def metadata(self) -> dict:
        """Session metadata (timestamps, model list, etc.)."""
        if self._metadata is None:
            self._metadata = self.repo.load_metadata()
        return self._metadata

    @property
    def class_mappings(self) -> dict:
        """Class label → integer encoding dict."""
        if self._class_mappings is None:
            self._class_mappings = self.repo.load_class_mappings()
        return self._class_mappings

    @property
    def survival(self) -> dict:
        """Survival analysis results (best_c_index, best_parameters, etc.)."""
        if self._survival is None:
            self._survival = self.repo.load_survival_results()
        return self._survival

    # ── Summary / quality ─────────────────────────────────────────────────────

    def summary(self) -> str:
        """Print and return a formatted summary of the current session."""
        df = self.df
        meta = self.metadata
        surv = self.survival

        best_idx = df["enhanced_f1_macro"].idxmax()
        best_row = df.loc[best_idx]

        lines = [
            "=" * 60,
            f"Session    : {self.current_session}",
            f"Experiments: {len(df):,}",
            f"Run start  : {meta.get('training_start_time', '?')}",
            f"Run end    : {meta.get('training_end_time', '?')}",
            f"Duration   : {meta.get('training_run_time', '?')}",
            "",
            f"Best F1 (enhanced) : {best_row['enhanced_f1_macro']:.4f}",
            f"Best AUC (enhanced): {best_row['enhanced_roc_auc_macro']:.4f}",
            f"Best model         : {best_row.get('model_display', best_row['model'])}",
            f"Best strategy      : {best_row.get('strategy_label', '')}",
            "",
            f"CoxPH C-index      : {surv.get('best_c_index', '?')}",
            "=" * 60,
        ]
        text = "\n".join(lines)
        print(text)
        return text

    def quality(self) -> QualityReport:
        """Run data quality checks and return a structured report."""
        return run_quality_checks(self.df)

    # ── Fluent query entry points ─────────────────────────────────────────────
    #
    # These create a ResultSet from the full DataFrame and immediately apply
    # the requested filter/ranking.  They're syntactic sugar for
    # ``ResultSet(self.df, self).method(...)``.

    def all(self) -> ResultSet:
        """Return a ResultSet containing all experiments."""
        return ResultSet(self.df, self)

    def top(self, n: int = 10, by: str = "enhanced_f1_macro") -> ResultSet:
        """Return the top-N experiments ranked by a metric."""
        return self.all().top(n, by=by)

    def best_per(self, group_col: str, by: str = "enhanced_f1_macro") -> ResultSet:
        """Return the best experiment per unique value of a column."""
        return self.all().best_per(group_col, by=by)

    def family(self, *families: str) -> ResultSet:
        """Filter to experiments in the given model families."""
        return self.all().family(*families)

    def model(self, *models: str) -> ResultSet:
        """Filter to experiments for specific model slugs."""
        return self.all().model(*models)

    def strategy(self, *strategies: str) -> ResultSet:
        """Filter to experiments using specific balance strategies."""
        return self.all().strategy(*strategies)

    def where(self, predicate=None, **kwargs) -> ResultSet:
        """Generic filter — see :meth:`ResultSet.where`."""
        return self.all().where(predicate, **kwargs)

    def variant_lift(self) -> pd.DataFrame:
        """Compare ordinal/two-stage variants against base classifiers."""
        return self.all().variant_lift()

    def rank_correlation(
        self,
        col_a: str = "enhanced_f1_macro",
        col_b: str = "enhanced_roc_auc_macro",
    ) -> ComparisonResult:
        """Spearman rank correlation between two metric columns."""
        return self.all().rank_correlation(col_a, col_b)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_df(self) -> pd.DataFrame:
        """
        Load the experiment summary and enrich with derived columns.
        """
        df = self.repo.load_experiments_summary()

        # Rename experiment_id → id to match notebook convention
        if "experiment_id" in df.columns and "id" not in df.columns:
            df = df.rename(columns={"experiment_id": "id"})

        # Derived columns
        df["strategy_label"] = df.apply(
            lambda r: _build_strategy_label(
                str(r["balance_strategy"]),
                r.get("undersample_threshold"),
            ),
            axis=1,
        )
        df["model_display"] = df["model"].map(MODEL_DISPLAY)
        df["family"] = df["model"].map(FAMILY_MAP)

        if "enhanced_f1_macro" in df.columns and "baseline_f1_macro" in df.columns:
            df["f1_lift"] = df["enhanced_f1_macro"] - df["baseline_f1_macro"]

        return df

    def __repr__(self) -> str:
        session = self.current_session or "(no session)"
        n = len(self.df) if self._df is not None else "?"
        return f"ResultsAnalyzer(session={session!r}, experiments={n})"
