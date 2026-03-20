# machine_learning/utils/io/migrate_db_schema.py
#
# SchemaV1Migrator
# ────────────────
# Converts an existing v1 flat-schema results.db (single wide "results" table)
# to the v2 normalized schema (experiments / metrics / charts / features +
# blob tables) used by ResultsRepository.
#
# The old "results" table is DROPPED after a successful migration — there is no
# backup and no backward-compatibility shim.  Run this once per database file;
# calling it on a database that is already on v2 is a safe no-op.
#
# CLI usage
# ─────────
#   python -m machine_learning.utils.io.migrate_db_schema Results/
#   python -m machine_learning.utils.io.migrate_db_schema Results/2025_03_05_01
#
# Programmatic usage
# ──────────────────
#   from machine_learning.utils.io.migrate_db_schema import SchemaV1Migrator
#
#   # Migrate a single database file
#   m = SchemaV1Migrator("Results/2025_03_05_01/results.db")
#   m.run()
#
#   # Migrate every session folder under a results root
#   SchemaV1Migrator.migrate_all("Results/")

from __future__ import annotations

import ast
import json
import os
import re
import sqlite3

import numpy as np

from .db_schema import ALL_DDL, SCHEMA_VERSION
from .results_repository import ResultsRepository


# ══════════════════════════════════════════════════════════════════════════════
#  V1 PARAM PARSING  (mirrors data_loaders.py exactly so old param strings
#  are decoded to clean dicts before being re-stored as proper JSON)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_two_stage_params(raw: str) -> dict | None:
    """
    Parse Two Stage param strings of the form::

        stage1={'key': val, ...}, stage2={'key': val, ...}

    Returns a nested ``{'stage1': {...}, 'stage2': {...}}`` dict, or ``None``
    when the pattern is not detected.
    """
    pattern = re.compile(r'(stage\w+)\s*=\s*(\{[^{}]*\})')
    matches = pattern.findall(raw)
    if not matches:
        return None
    result: dict = {}
    for stage_name, dict_str in matches:
        try:
            stage_dict = ast.literal_eval(dict_str)
            if isinstance(stage_dict, dict):
                result[stage_name] = stage_dict
        except Exception:
            pass
    return result if result else None


def _normalise_params(raw) -> dict:
    """
    Convert any v1 params representation to a clean ``{str: scalar}`` dict.

    Handles all four formats that appear in v1 databases:

    - Already a ``dict``
    - A list/tuple of 2-tuples: ``[('key', val), ...]``
    - A Two Stage string: ``"stage1={...}, stage2={...}"``
    - A single-quoted dict string: ``"{'key': val}"``
    """
    if isinstance(raw, dict):
        return raw

    if isinstance(raw, (list, tuple)):
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
        two_stage = _parse_two_stage_params(raw.strip())
        if two_stage is not None:
            return two_stage
        try:
            cleaned = (
                raw.strip()
                .replace("'", '"')
                .replace("True",  "true")
                .replace("False", "false")
                .replace("None",  "null")
            )
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        try:
            evaled = ast.literal_eval(raw.strip())
            return _normalise_params(evaled)
        except Exception:
            pass

    return {}


def _json_deserialize(value):
    """Deserialize a JSON string; return the original value on failure."""
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
#  MIGRATION RESULT
# ══════════════════════════════════════════════════════════════════════════════

class MigrationResult:
    """
    Value object returned by :meth:`SchemaV1Migrator.run`.

    Attributes
    ----------
    db_path : str
        The database file that was operated on.
    status : 'migrated' | 'already_v2' | 'no_db'
        Outcome of the migration attempt.
    rows_migrated : int
        Number of experiment rows written to the v2 schema.
        Zero when ``status != 'migrated'``.
    error : Exception or None
        Set when an unhandled exception occurred during migration.
    """

    __slots__ = ("db_path", "status", "rows_migrated", "error")

    def __init__(
        self,
        db_path: str,
        status: str,
        rows_migrated: int = 0,
        error: Exception | None = None,
    ) -> None:
        self.db_path       = db_path
        self.status        = status
        self.rows_migrated = rows_migrated
        self.error         = error

    def __repr__(self) -> str:  # pragma: no cover
        if self.error:
            return (
                f"MigrationResult(status={self.status!r}, "
                f"db={self.db_path!r}, error={self.error!r})"
            )
        return (
            f"MigrationResult(status={self.status!r}, "
            f"rows_migrated={self.rows_migrated}, db={self.db_path!r})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MIGRATOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

class SchemaV1Migrator:
    """
    Converts a v1 flat-schema ``results.db`` to the v2 normalized schema.

    The v1 schema has a single wide ``results`` table where every experiment
    row contains scalar metrics, chart blobs, and feature lists all in the
    same row.  The v2 schema splits those concerns into four tables
    (``experiments``, ``metrics``, ``charts``, ``features``) so that the
    dashboard leaderboard can load instantly without deserializing chart blobs.

    After a successful migration the old ``results`` table is **dropped**.
    This is intentional — the v2 schema is not backward compatible, and
    keeping the stale table would waste disk space and cause confusion.

    Parameters
    ----------
    db_path : str
        Absolute or relative path to a ``results.db`` file.

    Examples
    --------
    Migrate one file::

        m = SchemaV1Migrator("Results/2025_03_05_01/results.db")
        result = m.run()
        print(result)

    Migrate every session under a results root::

        results = SchemaV1Migrator.migrate_all("Results/")
        for r in results:
            print(r)
    """

    # Matches YYYY_MM_DD_## session folder names.
    _DATE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}$")

    def __init__(self, db_path: str) -> None:
        self.db_path = os.path.abspath(db_path)

    # ── Public entry points ───────────────────────────────────────────────────

    def run(self) -> MigrationResult:
        """
        Execute the migration for this database file.

        Returns a :class:`MigrationResult` describing the outcome.  Never
        raises — all exceptions are caught and stored in
        ``MigrationResult.error`` so that :meth:`migrate_all` can continue
        past a single bad file.
        """
        if not os.path.exists(self.db_path):
            print(f"[migrate] SKIP  (file not found): {self.db_path}")
            return MigrationResult(self.db_path, status="no_db")

        if not self._is_v1():
            print(f"[migrate] SKIP  (already v2):     {self.db_path}")
            return MigrationResult(self.db_path, status="already_v2")

        print(f"[migrate] START migrating:         {self.db_path}")
        try:
            rows = self._migrate()
            print(f"[migrate] DONE  ({rows} rows):      {self.db_path}")
            return MigrationResult(self.db_path, status="migrated", rows_migrated=rows)
        except Exception as exc:
            print(f"[migrate] ERROR:                   {self.db_path}\n  {exc}")
            return MigrationResult(self.db_path, status="migrated", error=exc)

    @classmethod
    def migrate_all(
        cls,
        results_root: str,
        *,
        dry_run: bool = False,
    ) -> list[MigrationResult]:
        """
        Discover every dated session folder under *results_root* and migrate
        each ``results.db`` found there.

        Parameters
        ----------
        results_root : str
            Root directory that contains ``YYYY_MM_DD_##`` session folders.
        dry_run : bool, default False
            When ``True``, print what *would* be migrated but make no changes.

        Returns
        -------
        list[MigrationResult]
            One entry per session folder found, regardless of outcome.
        """
        results_root = os.path.abspath(results_root)
        if not os.path.isdir(results_root):
            raise FileNotFoundError(
                f"Results root not found: {results_root!r}"
            )

        session_dirs = sorted(
            [
                d for d in os.listdir(results_root)
                if cls._DATE_RE.match(d)
                and os.path.isdir(os.path.join(results_root, d))
            ],
            reverse=True,
        )

        if not session_dirs:
            print(f"[migrate] No dated session folders found under {results_root!r}")
            return []

        print(
            f"[migrate] Found {len(session_dirs)} session folder(s) under "
            f"{results_root!r}"
        )

        outcomes: list[MigrationResult] = []
        for folder in session_dirs:
            db_path = os.path.join(results_root, folder, "results.db")
            if dry_run:
                exists = os.path.exists(db_path)
                print(
                    f"[migrate] DRY-RUN {'(would migrate)' if exists else '(no db)':20s}  "
                    f"{db_path}"
                )
                outcomes.append(
                    MigrationResult(
                        db_path,
                        status="migrated" if exists else "no_db",
                    )
                )
            else:
                outcomes.append(cls(db_path).run())

        return outcomes

    # ── Schema detection ──────────────────────────────────────────────────────

    def _is_v1(self) -> bool:
        """
        Return ``True`` when the database has the old flat ``results`` table
        but not the v2 ``experiments`` table.
        """
        with sqlite3.connect(self.db_path) as con:
            tables = {
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        return "results" in tables and "experiments" not in tables

    # ── Core migration logic ──────────────────────────────────────────────────

    def _migrate(self) -> int:
        """
        Read v1 data, write v2 schema, drop v1 table.

        Returns the number of experiment rows written.
        """
        v1_rows, class_mappings, survival_results, metadata = self._read_v1()

        repo = ResultsRepository(self.db_path)
        repo.initialize_schema()

        with repo._connect() as con:
            for row in v1_rows:
                exp_id = self._write_experiment(con, row)
                for phase in ("baseline", "enhanced"):
                    self._write_metrics(con, exp_id, phase, row)
                    self._write_charts(con, exp_id, phase, row)
                    self._write_features(con, exp_id, phase, row)

            repo._upsert_blob(con, "class_mappings",   class_mappings)
            repo._upsert_blob(con, "survival_results", survival_results)
            repo._upsert_blob(con, "metadata",         metadata)

            # Drop the old flat table — no backward compatibility shim.
            con.execute("DROP TABLE IF EXISTS results")
            print(f"[migrate]   Dropped legacy 'results' table from {self.db_path}")

        return len(v1_rows)

    # ── V1 readers ────────────────────────────────────────────────────────────

    def _read_v1(self) -> tuple[list[dict], dict, dict, dict]:
        """
        Read all data from the v1 flat schema.

        Returns
        -------
        (rows, class_mappings, survival_results, metadata)
        """
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            cur = con.execute("SELECT * FROM results")
            columns = [d[0] for d in cur.description]
            raw_rows = []
            for sqlite_row in cur.fetchall():
                row: dict = {}
                for col in columns:
                    row[col] = _json_deserialize(sqlite_row[col])
                raw_rows.append(row)

            class_mappings   = self._read_blob(con, "class_mappings")
            survival_results = self._read_blob(con, "survival_results")
            metadata         = self._read_blob(con, "metadata")

        print(
            f"[migrate]   Read {len(raw_rows)} row(s) from v1 schema, "
            f"{len(columns)} column(s)"
        )
        return raw_rows, class_mappings, survival_results, metadata

    @staticmethod
    def _read_blob(con: sqlite3.Connection, table: str) -> dict:
        try:
            row = con.execute(
                f"SELECT data FROM {table} WHERE id=1"
            ).fetchone()
            return _json_deserialize(row[0]) if row else {}
        except sqlite3.OperationalError:
            return {}

    # ── V2 writers ────────────────────────────────────────────────────────────

    @staticmethod
    def _get(row: dict, col: str):
        v = row.get(col)
        return v if v is not None else ""

    def _write_experiment(self, con: sqlite3.Connection, row: dict) -> int:
        raw_params = _json_deserialize(self._get(row, "parameters"))
        params     = _normalise_params(raw_params)
        repo       = ResultsRepository.__new__(ResultsRepository)
        p_hash     = ResultsRepository._param_hash(params)
        p_json     = ResultsRepository._to_json(params)

        cur = con.execute(
            """
            INSERT INTO experiments (model, balance_strategy, parameters, param_hash)
            VALUES (?, ?, ?, ?)
            """,
            (
                str(self._get(row, "model")),
                str(self._get(row, "balance_strategy") or "none"),
                p_json,
                p_hash,
            ),
        )
        return cur.lastrowid

    @staticmethod
    def _write_metrics(
        con: sqlite3.Connection, exp_id: int, phase: str, row: dict
    ) -> None:
        def _f(col):
            v = row.get(f"{phase}_{col}")
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        con.execute(
            """
            INSERT INTO metrics
                (experiment_id, phase, accuracy, precision_macro,
                 recall_macro, f1_macro, roc_auc_macro)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exp_id, phase,
                _f("accuracy"), _f("precision_macro"),
                _f("recall_macro"), _f("f1_macro"), _f("roc_auc_macro"),
            ),
        )

    @staticmethod
    def _write_charts(
        con: sqlite3.Connection, exp_id: int, phase: str, row: dict
    ) -> None:
        for chart_type in ("confusion_matrix", "roc_curve", "pr_curve"):
            raw = row.get(f"{phase}_{chart_type}")
            if raw is None:
                continue
            data_str = (
                raw if isinstance(raw, str)
                else ResultsRepository._to_json(raw)
            )
            con.execute(
                """
                INSERT INTO charts (experiment_id, phase, chart_type, data)
                VALUES (?, ?, ?, ?)
                """,
                (exp_id, phase, chart_type, data_str),
            )

    @staticmethod
    def _write_features(
        con: sqlite3.Connection, exp_id: int, phase: str, row: dict
    ) -> None:
        raw_selected = row.get(f"{phase}_feature_selected")
        features: list = (
            raw_selected if isinstance(raw_selected, list)
            else (_json_deserialize(raw_selected) if raw_selected else [])
        )

        raw_weights = row.get(f"{phase}_feature_weights")
        weights = (
            raw_weights if isinstance(raw_weights, (list, dict))
            else (_json_deserialize(raw_weights) if raw_weights else None)
        )

        raw_fp = row.get(f"{phase}_feature_parameters")
        feat_params = (
            raw_fp if isinstance(raw_fp, dict)
            else (_json_deserialize(raw_fp) if raw_fp else None)
        )

        con.execute(
            """
            INSERT INTO features
                (experiment_id, phase, feature_method, feature_parameters,
                 features_json, weights_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                exp_id,
                phase,
                str(row.get(f"{phase}_feature_method") or "").strip(),
                ResultsRepository._to_json(feat_params) if feat_params else None,
                ResultsRepository._to_json(features),
                ResultsRepository._to_json(weights) if weights is not None else None,
            ),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _cli() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="migrate_db_schema",
        description=(
            "Migrate results.db file(s) from the v1 flat schema to the "
            "v2 normalized schema used by ResultsRepository."
        ),
    )
    parser.add_argument(
        "path",
        help=(
            "Path to a results root folder (e.g. Results/) to migrate all "
            "session sub-folders, OR a direct path to a single results.db file."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be migrated without making any changes.",
    )
    args = parser.parse_args()
    target = os.path.abspath(args.path)

    # Single .db file
    if target.endswith(".db"):
        result = SchemaV1Migrator(target).run()
        sys.exit(0 if result.error is None else 1)

    # Results root folder
    outcomes = SchemaV1Migrator.migrate_all(target, dry_run=args.dry_run)
    errors   = [r for r in outcomes if r.error is not None]
    migrated = [r for r in outcomes if r.status == "migrated" and r.error is None]
    skipped  = [r for r in outcomes if r.status in ("already_v2", "no_db")]

    print(
        f"\n[migrate] Summary: "
        f"{len(migrated)} migrated, "
        f"{len(skipped)} skipped, "
        f"{len(errors)} error(s)"
    )
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    _cli()
