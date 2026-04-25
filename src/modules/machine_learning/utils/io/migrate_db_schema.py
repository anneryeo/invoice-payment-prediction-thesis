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
#   python -m machine_learning.utils.io.migrate_db_schema "Results - Backup/" "results/"
#   python -m machine_learning.utils.io.migrate_db_schema "Results - Backup/" "results/" --keep-db
#   python -m machine_learning.utils.io.migrate_db_schema results/2025_03_05_01/results.db
#
# Programmatic usage
# ──────────────────
#  from src.modules.machine_learning.utils.io.migrate_db_schema import SchemaV1Migrator
#
#   # Migrate a single database file (in-place)
#   m = SchemaV1Migrator("Results/2025_03_05_01/results.db")
#   m.run()
#
#   # Migrate every session folder from source into dest
#   SchemaV1Migrator.migrate_all(
#       "ignored",
#       source_folder="Results - Backup",
#       dest_folder="Results",
#   )

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
        Absolute or relative path to the **destination** ``results.db`` file
        that will be migrated in-place.  When using :meth:`migrate_all` with
        a ``dest_folder``, this is the copy inside the destination folder —
        the original source file is never mutated.
    keep_db : bool, default False
        When ``True``, copy ``results.db`` to ``results.db.bak`` before
        making any changes.

    Examples
    --------
    Migrate one file in-place::

        m = SchemaV1Migrator("results/2025_03_05_01/results.db")
        result = m.run()
        print(result)

    Migrate every session from a backup folder into a fresh results folder::

        results = SchemaV1Migrator.migrate_all(
            "ignored",
            source_folder="Results - Backup",
            dest_folder="Results",
        )
        for r in results:
            print(r)
    """

    # Matches YYYY_MM_DD_## session folder names.
    _DATE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}$")

    def __init__(self, db_path: str, *, keep_db: bool = False) -> None:
        self.db_path = os.path.abspath(db_path)
        self.keep_db = keep_db

    # ── Public entry points ───────────────────────────────────────────────────

    def run(self) -> MigrationResult:
        """
        Execute the migration for this database file.

        When ``keep_db`` is ``True`` the original ``results.db`` is copied to
        ``results.db.bak`` before any changes are made, preserving the v1 data.

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

        if self.keep_db:
            import shutil
            backup_path = self.db_path + ".bak"
            shutil.copy2(self.db_path, backup_path)
            print(f"[migrate] BACKUP created:          {backup_path}")

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
        keep_db: bool = False,
        source_folder: str | None = None,
        dest_folder: str | None = None,
    ) -> list[MigrationResult]:
        """
        Discover every dated session folder under *source_folder* and migrate
        each ``results.db`` into the corresponding sub-folder of *dest_folder*.

        Parameters
        ----------
        results_root : str
            Fallback root directory used when *source_folder* is not provided.
        dry_run : bool, default False
            When ``True``, print what *would* be migrated but make no changes.
        keep_db : bool, default False
            When ``True``, copy each ``results.db`` to ``results.db.bak``
            before migrating so the original v1 data is preserved.
        source_folder : str
            Path to the original results folder containing ``YYYY_MM_DD_##``
            session sub-folders (e.g. ``"Results - Backup"``).
        dest_folder : str
            Path to the destination results folder where migrated databases
            will be written (e.g. ``"Results"``).  Session sub-folders are
            mirrored from *source_folder* and created if they do not exist.

        Returns
        -------
        list[MigrationResult]
            One entry per session folder found, regardless of outcome.
        """
        import shutil

        # Resolve source (where we read session folders from).
        effective_source = os.path.abspath(
            source_folder if source_folder is not None else results_root
        )
        if not os.path.isdir(effective_source):
            raise FileNotFoundError(
                f"Source folder not found: {effective_source!r}"
            )

        # Resolve destination (where migrated DBs are written).
        effective_dest = (
            os.path.abspath(dest_folder) if dest_folder is not None else None
        )
        if effective_dest is not None and not dry_run:
            os.makedirs(effective_dest, exist_ok=True)

        session_dirs = sorted(
            [
                d for d in os.listdir(effective_source)
                if cls._DATE_RE.match(d)
                and os.path.isdir(os.path.join(effective_source, d))
            ],
            reverse=True,
        )

        if not session_dirs:
            print(f"[migrate] No dated session folders found under {effective_source!r}")
            return []

        print(
            f"[migrate] Found {len(session_dirs)} session folder(s) under "
            f"{effective_source!r}"
        )
        if effective_dest is not None:
            print(f"[migrate] Destination folder: {effective_dest!r}")

        outcomes: list[MigrationResult] = []
        for folder in session_dirs:
            src_db = os.path.join(effective_source, folder, "results.db")

            if effective_dest is not None:
                # dest_db is the copy inside the destination — this is what
                # gets migrated; the source file is never touched.
                dest_session = os.path.join(effective_dest, folder)
                dest_db = os.path.join(dest_session, "results.db")
                if not dry_run and os.path.exists(src_db) and not os.path.exists(dest_db):
                    os.makedirs(dest_session, exist_ok=True)
                    shutil.copy2(src_db, dest_db)
                    print(f"[migrate] COPIED source DB to:     {dest_db}")
                migrate_db = dest_db
            else:
                # No dest_folder supplied — migrate in-place inside source.
                migrate_db = src_db

            if dry_run:
                exists = os.path.exists(src_db)
                print(
                    f"[migrate] DRY-RUN {'(would migrate)' if exists else '(no db)':20s}  "
                    f"{migrate_db}"
                )
                outcomes.append(
                    MigrationResult(
                        migrate_db,
                        status="migrated" if exists else "no_db",
                    )
                )
            else:
                outcomes.append(cls(migrate_db, keep_db=keep_db).run())

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
        # Preserve raw weight values exactly — do NOT normalise or rescale.
        # _json_deserialize has already been applied to every column in
        # _read_v1, so raw_weights is already a Python list/dict/scalar or
        # the original string if parsing failed.  Convert numpy arrays to
        # plain Python lists so _to_json round-trips without precision loss.
        if isinstance(raw_weights, np.ndarray):
            weights = raw_weights.tolist()
        elif isinstance(raw_weights, (list, dict)):
            weights = raw_weights
        elif raw_weights is not None and raw_weights != "":
            weights = _json_deserialize(raw_weights)
        else:
            weights = None

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
        "source",
        help=(
            'Path to the source results folder containing YYYY_MM_DD_## session '
            'sub-folders (e.g. "Results - Backup/"), OR a direct path to a '
            "single results.db file (migrated in-place, no dest needed)."
        ),
    )
    parser.add_argument(
        "dest",
        nargs="?",
        default=None,
        help=(
            'Path to the destination results folder where migrated databases '
            'will be written (e.g. "results/"). Session sub-folders are mirrored '
            "from source and created if they do not exist. Required when source "
            "is a folder; omit only when source is a single .db file."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be migrated without making any changes.",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        default=False,
        help=(
            "Preserve the original results.db by copying it to results.db.bak "
            "before any changes are made. Has no effect on --dry-run."
        ),
    )
    args = parser.parse_args()
    source = os.path.abspath(args.source)

    # Single .db file — in-place migration, dest is not applicable.
    if source.endswith(".db"):
        result = SchemaV1Migrator(source, keep_db=args.keep_db).run()
        sys.exit(0 if result.error is None else 1)

    # Folder migration — dest is required.
    if args.dest is None:
        parser.error(
            "A destination folder is required when source is a results root.\n"
            '  Example: migrate_db_schema "Results - Backup/" "results/"'
        )

    outcomes = SchemaV1Migrator.migrate_all(
        source,
        dry_run=args.dry_run,
        keep_db=args.keep_db,
        source_folder=args.source,
        dest_folder=args.dest,
    )
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
