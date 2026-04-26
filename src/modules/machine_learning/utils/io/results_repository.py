# machine_learning/utils/io/results_repository.py
#
# ResultsRepository — single OOP interface for all SQLite I/O.
#
# Design goals
# ────────────
#   • Leaderboard / summary data loads in milliseconds via a JOIN-only query
#     that never touches the charts table.
#   • Heavy chart blobs (roc_curve, pr_curve, confusion_matrix) are fetched
#     only when a user explicitly opens the model-detail modal.
#   • All DDL lives in db_schema.py; ResultsRepository never hard-codes SQL
#     table or column names that would drift out of sync.
#   • Backward-compatible: detects old flat-schema databases and migrates them
#     transparently on first access.

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from typing import Optional

import numpy as np
import pandas as pd

from .db_schema import ALL_DDL, SCHEMA_VERSION


# ══════════════════════════════════════════════════════════════════════════════
#  ResultsRepository
# ══════════════════════════════════════════════════════════════════════════════

class ResultsRepository:
    """
    OOP interface for reading and writing experiment results to a SQLite
    database that uses the normalized v2 schema.

    Normalized schema
    -----------------
    experiments      – lightweight run identifiers (model, strategy, params)
    metrics          – scalar evaluation metrics (accuracy, f1, …)
    charts           – heavy JSON blobs (roc_curve, pr_curve, confusion_matrix)
    features         – feature selection lists and weights
    class_mappings   – single-row JSON blob
    survival_results – single-row JSON blob
    metadata         – single-row JSON blob

    Parameters
    ----------
    db_path : str
        Absolute path to ``results.db``.

    Examples
    --------
    Saving a new session::

        repo = ResultsRepository(db_path)
        repo.save_session(model_results_df, class_mappings,
                          survival_results, metadata)

    Fast leaderboard load (no chart data)::

        repo = ResultsRepository(db_path)
        models = repo.load_models_dict()

    On-demand chart hydration when a user opens a model modal::

        repo.hydrate_model_charts(models[selected_key])
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    # ── Connection context manager ────────────────────────────────────────────

    @contextmanager
    def _connect(self):
        """
        Yield a committed, auto-rollback SQLite connection.

        WAL mode is enabled so concurrent readers (Dash callbacks) never
        block the single writer (save_session).
        """
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode  = WAL")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    # ── Schema management ─────────────────────────────────────────────────────

    def initialize_schema(self) -> None:
        """
        Create all tables and indexes if they do not yet exist, then record
        the current schema version.  Safe to call repeatedly (idempotent).
        """
        with self._connect() as con:
            for ddl in ALL_DDL:
                con.execute(ddl)
            row = con.execute(
                "SELECT version FROM schema_version WHERE id=1"
            ).fetchone()
            if row is None:
                con.execute(
                    "INSERT INTO schema_version (id, version) VALUES (1, ?)",
                    (SCHEMA_VERSION,),
                )
            else:
                db_version = row[0]
                # v2 → v3: add undersample_threshold column if missing
                if db_version < 3:
                    existing = {r[1] for r in con.execute(
                        "PRAGMA table_info(experiments)"
                    ).fetchall()}
                    if "undersample_threshold" not in existing:
                        con.execute(
                            "ALTER TABLE experiments ADD COLUMN undersample_threshold REAL"
                        )
                
                # v3 → v4: add cache_key column and create cache_registry table
                if db_version < 4:
                    existing = {r[1] for r in con.execute(
                        "PRAGMA table_info(experiments)"
                    ).fetchall()}
                    if "cache_key" not in existing:
                        con.execute(
                            "ALTER TABLE experiments ADD COLUMN cache_key TEXT"
                        )
                    
                    con.execute(
                        "UPDATE schema_version SET version=4 WHERE id=1"
                    )

    # ── Serialization helpers ─────────────────────────────────────────────────

    @staticmethod
    def _to_json(obj) -> str:
        """Recursively convert numpy types and serialize to a JSON string."""
        def _sanitize(o):
            if isinstance(o, dict):
                return {k: _sanitize(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_sanitize(i) for i in o]
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o
        return json.dumps(_sanitize(obj))

    @staticmethod
    def _from_json(text):
        """
        Deserialize a JSON string.  Returns the original value unchanged
        when it is not a string or is not valid JSON.
        """
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return text
        if isinstance(text, (dict, list)):
            return text
        if not isinstance(text, str):
            return text
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text

    @staticmethod
    def _param_hash(params: dict) -> str:
        """Return a 6-character MD5 hex digest of the serialized parameter dict."""
        sig = json.dumps(params, sort_keys=True) if params else "default"
        return hashlib.md5(sig.encode()).hexdigest()[:6]

    # ══════════════════════════════════════════════════════════════════════════
    #  WRITE
    # ══════════════════════════════════════════════════════════════════════════

    def save_session(
        self,
        model_results_df: pd.DataFrame,
        class_mappings: dict,
        survival_results: dict,
        metadata: dict,
    ) -> None:
        """
        Persist a complete training session to the normalized SQLite schema.
        """
        self.initialize_schema()

        with self._connect() as con:
            self._upsert_blob(con, "class_mappings",   class_mappings)
            self._upsert_blob(con, "survival_results", survival_results)
            self._upsert_blob(con, "metadata",         metadata)

            for _, row in model_results_df.iterrows():
                exp_id = self._insert_experiment(con, row)
                for phase in ("baseline", "enhanced"):
                    self._insert_metrics(con, exp_id, phase, row)
                    self._insert_charts(con, exp_id, phase, row)
                    self._insert_features(con, exp_id, phase, row)

    # ── Private write helpers ─────────────────────────────────────────────────

    def _upsert_blob(self, con: sqlite3.Connection, table: str, data: dict) -> None:
        con.execute(f"DELETE FROM {table}")
        con.execute(
            f"INSERT INTO {table} (id, data) VALUES (1, ?)",
            (self._to_json(data),),
        )

    def _insert_experiment(self, con: sqlite3.Connection, row) -> int:
        """Insert one experiments row and return the new primary key."""
        raw_params = row.get("parameters")
        params: dict = (
            self._from_json(raw_params)
            if isinstance(raw_params, str)
            else (raw_params if isinstance(raw_params, dict) else {})
        )
        p_hash = self._param_hash(params)
        threshold = row.get("undersample_threshold")
        try:
            threshold = float(threshold) if threshold is not None else None
        except (TypeError, ValueError):
            threshold = None
        
        cur = con.execute(
            """
            INSERT INTO experiments
                (model, balance_strategy, undersample_threshold, parameters, param_hash, cache_key)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(row.get("model", "")),
                str(row.get("balance_strategy", "none")),
                threshold,
                self._to_json(params),
                p_hash,
                row.get("cache_key"),
            ),
        )
        return cur.lastrowid

    def _insert_metrics(
        self, con: sqlite3.Connection, exp_id: int, phase: str, row
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

    def _insert_charts(
        self, con: sqlite3.Connection, exp_id: int, phase: str, row
    ) -> None:
        for chart_type in ("confusion_matrix", "roc_curve", "pr_curve"):
            raw = row.get(f"{phase}_{chart_type}")
            if raw is None:
                continue
            # Serialize if not already a string (e.g. still a dict/list).
            data_str = raw if isinstance(raw, str) else self._to_json(raw)
            con.execute(
                """
                INSERT INTO charts (experiment_id, phase, chart_type, data)
                VALUES (?, ?, ?, ?)
                """,
                (exp_id, phase, chart_type, data_str),
            )

    def _insert_features(
        self, con: sqlite3.Connection, exp_id: int, phase: str, row
    ) -> None:
        raw_selected = row.get(f"{phase}_feature_selected")
        features: list = (
            raw_selected if isinstance(raw_selected, list)
            else (self._from_json(raw_selected) if raw_selected else [])
        )

        raw_weights = row.get(f"{phase}_feature_weights")
        weights = (
            raw_weights if isinstance(raw_weights, (list, dict))
            else (self._from_json(raw_weights) if raw_weights else None)
        )

        raw_params = row.get(f"{phase}_feature_parameters")
        feat_params = (
            raw_params if isinstance(raw_params, dict)
            else (self._from_json(raw_params) if raw_params else None)
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
                self._to_json(feat_params) if feat_params else None,
                self._to_json(features),
                self._to_json(weights) if weights is not None else None,
            ),
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  READ — lightweight (no charts)
    # ══════════════════════════════════════════════════════════════════════════

    def load_experiments_summary(self) -> pd.DataFrame:
        """
        Return all experiments with their metric scalars but **without** any
        chart data.
        """
        sql = """
            SELECT
                e.id                    AS experiment_id,
                e.model,
                e.balance_strategy,
                e.undersample_threshold,
                e.parameters,
                e.param_hash,
                e.cache_key,
                b.accuracy              AS baseline_accuracy,
                b.precision_macro  AS baseline_precision_macro,
                b.recall_macro     AS baseline_recall_macro,
                b.f1_macro         AS baseline_f1_macro,
                b.roc_auc_macro    AS baseline_roc_auc_macro,
                n.accuracy         AS enhanced_accuracy,
                n.precision_macro  AS enhanced_precision_macro,
                n.recall_macro     AS enhanced_recall_macro,
                n.f1_macro         AS enhanced_f1_macro,
                n.roc_auc_macro    AS enhanced_roc_auc_macro
            FROM  experiments e
            LEFT JOIN metrics b
                   ON b.experiment_id = e.id AND b.phase = 'baseline'
            LEFT JOIN metrics n
                   ON n.experiment_id = e.id AND n.phase = 'enhanced'
            ORDER BY e.id
        """
        with sqlite3.connect(self.db_path) as con:
            df = pd.read_sql(sql, con)

        df["parameters"] = df["parameters"].apply(self._from_json)
        return df

    # ══════════════════════════════════════════════════════════════════════════
    #  READ — on-demand (charts / features)
    # ══════════════════════════════════════════════════════════════════════════

    def load_charts(
        self,
        experiment_id: int,
        phase: str,
        chart_type: Optional[str] = None,
    ) -> dict:
        """
        Fetch chart blobs for a single experiment on demand.
        """
        sql = """
            SELECT chart_type, data
            FROM   charts
            WHERE  experiment_id = ? AND phase = ?
        """
        bind: tuple = (experiment_id, phase)
        if chart_type is not None:
            sql += " AND chart_type = ?"
            bind = (experiment_id, phase, chart_type)

        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(sql, bind).fetchall()

        return {r[0]: self._from_json(r[1]) for r in rows}

    def load_features(self, experiment_id: int, phase: str) -> dict:
        """
        Fetch feature selection results for a single experiment on demand.
        """
        sql = """
            SELECT feature_method, feature_parameters, features_json, weights_json
            FROM   features
            WHERE  experiment_id = ? AND phase = ?
        """
        with sqlite3.connect(self.db_path) as con:
            row = con.execute(sql, (experiment_id, phase)).fetchone()

        if not row:
            return {
                "feature_method":     "",
                "feature_parameters": None,
                "features":           [],
                "weights":            None,
            }
        return {
            "feature_method":     row[0] or "",
            "feature_parameters": self._from_json(row[1]),
            "features":           self._from_json(row[2]) or [],
            "weights":            self._from_json(row[3]),
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  READ — blob tables
    # ══════════════════════════════════════════════════════════════════════════

    def load_metadata(self) -> dict:
        """Return the metadata dict for this session."""
        return self._load_blob("metadata")

    def load_class_mappings(self) -> dict:
        """Return the class-label → integer encoding dict."""
        return self._load_blob("class_mappings")

    def load_survival_results(self) -> dict:
        """Return the survival analysis results dict."""
        return self._load_blob("survival_results")

    def _load_blob(self, table: str) -> dict:
        with sqlite3.connect(self.db_path) as con:
            row = con.execute(
                f"SELECT data FROM {table} WHERE id=1"
            ).fetchone()
        return self._from_json(row[0]) if row else {}

    # ══════════════════════════════════════════════════════════════════════════
    #  READ — MODELS-compatible dict
    # ══════════════════════════════════════════════════════════════════════════

    def load_models_dict(self) -> dict:
        """
        Build the ``MODELS``-compatible dict used by the Dash dashboard.
        """
        df = self.load_experiments_summary()
        models: dict = {}

        for _, row in df.iterrows():
            params   = row["parameters"] if isinstance(row["parameters"], dict) else {}
            p_hash   = row["param_hash"] or self._param_hash(params)
            slug     = re.sub(r"\s+", "_", str(row["model"])).lower()
            key      = f"{slug}__{row['balance_strategy']}__{p_hash}"

            models[key] = {
                "experiment_id":    int(row["experiment_id"]),
                "model":            str(row["model"]),
                "balance_strategy": str(row["balance_strategy"]),
                "parameters":       params,
                "baseline":         self._build_section(row, "baseline"),
                "enhanced":         self._build_section(row, "enhanced"),
            }

        return models

    @staticmethod
    def _build_section(row, phase: str) -> dict:
        """Construct a baseline/enhanced section dict with None chart placeholders."""
        p = f"{phase}_"
        metrics = {
            "accuracy":        float(row.get(f"{p}accuracy")        or 0.0),
            "precision_macro": float(row.get(f"{p}precision_macro") or 0.0),
            "recall_macro":    float(row.get(f"{p}recall_macro")    or 0.0),
            "f1_macro":        float(row.get(f"{p}f1_macro")        or 0.0),
            "roc_auc_macro":   float(row.get(f"{p}roc_auc_macro")   or 0.0),
        }
        charts = {
            "confusion_matrix": None,
            "roc_curve":        None,
            "pr_curve":         None,
        }
        return {
            "evaluation":     {"metrics": metrics, "charts": charts},
            "features":       [],
            "feature_method": "",
        }

    def hydrate_model_charts(self, model_entry: dict) -> dict:
        """
        Fetch chart and feature data for a single model entry and inject it
        in-place.
        """
        exp_id = model_entry["experiment_id"]
        for phase in ("baseline", "enhanced"):
            charts = self.load_charts(exp_id, phase)
            model_entry[phase]["evaluation"]["charts"].update(charts)

            feat = self.load_features(exp_id, phase)
            model_entry[phase]["features"]           = feat["features"]
            model_entry[phase]["feature_method"]     = feat["feature_method"]
            model_entry[phase]["feature_parameters"] = feat.get("feature_parameters")
            model_entry[phase]["weights"]            = feat.get("weights")

        return model_entry

    # ══════════════════════════════════════════════════════════════════════════
    #  READ — backward-compat full DataFrame (for legacy callers)
    # ══════════════════════════════════════════════════════════════════════════

    def load_as_flat_dataframe(self) -> pd.DataFrame:
        """
        Reconstruct a flat DataFrame in the original v1 column layout.
        """
        df = self.load_experiments_summary()
        if df.empty:
            return df

        chart_cols:   dict[str, dict] = {}  # exp_id → {phase_charttype: data}
        feature_cols: dict[str, dict] = {}

        with sqlite3.connect(self.db_path) as con:
            for row in con.execute(
                "SELECT experiment_id, phase, chart_type, data FROM charts"
            ).fetchall():
                exp_id, phase, ctype, data = row
                chart_cols.setdefault(exp_id, {})[f"{phase}_{ctype}"] = (
                    self._from_json(data)
                )

            for row in con.execute(
                "SELECT experiment_id, phase, feature_method, "
                "feature_parameters, features_json, weights_json FROM features"
            ).fetchall():
                exp_id, phase, method, fp, fj, wj = row
                feature_cols.setdefault(exp_id, {}).update({
                    f"{phase}_feature_method":     method or "",
                    f"{phase}_feature_parameters": self._from_json(fp),
                    f"{phase}_feature_selected":   self._from_json(fj) or [],
                    f"{phase}_feature_weights":    self._from_json(wj),
                })

        extra_rows = []
        for _, row in df.iterrows():
            exp_id = int(row["experiment_id"])
            merged = row.to_dict()
            merged.update(chart_cols.get(exp_id, {}))
            merged.update(feature_cols.get(exp_id, {}))
            extra_rows.append(merged)

        return pd.DataFrame(extra_rows)

    # ══════════════════════════════════════════════════════════════════════════
    #  CACHING REGISTRY
    # ══════════════════════════════════════════════════════════════════════════

    def register_cache_item(
        self,
        cache_key: str,
        cache_type: str,
        parameters_hash: str,
        file_path: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Record a cached dataset or model in the central registry.
        """
        sql = """
            INSERT OR REPLACE INTO cache_registry
                (cache_key, cache_type, parameters_hash, file_path, metadata)
            VALUES (?, ?, ?, ?, ?)
        """
        with self._connect() as con:
            con.execute(
                sql,
                (
                    cache_key,
                    cache_type,
                    parameters_hash,
                    file_path,
                    self._to_json(metadata) if metadata else None,
                ),
            )

    def get_cache_entry(self, cache_key: str) -> Optional[dict]:
        """
        Retrieve registration details for a specific cache key.
        """
        sql = "SELECT * FROM cache_registry WHERE cache_key = ?"
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            row = con.execute(sql, (cache_key,)).fetchone()
        
        if row:
            d = dict(row)
            d["metadata"] = self._from_json(d["metadata"])
            return d
        return None
