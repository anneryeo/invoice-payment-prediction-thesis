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
        Persist a complete training session to the normalized v2 schema.

        Each row of *model_results_df* is decomposed into:
        - one ``experiments`` row  (model identity)
        - two ``metrics`` rows     (one per phase)
        - up to six ``charts`` rows (three chart types × two phases)
        - two ``features`` rows    (one per phase)

        The three blob tables (class_mappings, survival_results, metadata)
        are always written as a single replace-all-rows operation.

        Parameters
        ----------
        model_results_df : pd.DataFrame
            Flat experiment results as produced by SurvivalExperimentRunner.run().
            Expected column layout::

                model, balance_strategy, parameters,
                baseline_accuracy, baseline_precision_macro, baseline_recall_macro,
                baseline_f1_macro, baseline_roc_auc_macro,
                baseline_confusion_matrix, baseline_roc_curve, baseline_pr_curve,
                baseline_feature_method, baseline_feature_parameters,
                baseline_feature_selected, baseline_feature_weights,
                enhanced_*  (same set)

        class_mappings : dict
            Original class label → encoded integer mapping.
        survival_results : dict
            Best survival analysis results (c_index, params, time_points).
        metadata : dict
            Run metadata (timestamps, model list, …).
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
        cur = con.execute(
            """
            INSERT INTO experiments (model, balance_strategy, parameters, param_hash)
            VALUES (?, ?, ?, ?)
            """,
            (
                str(row.get("model", "")),
                str(row.get("balance_strategy", "none")),
                self._to_json(params),
                p_hash,
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

        The underlying SQL query joins ``experiments`` with ``metrics`` twice
        (once for baseline, once for enhanced) and never touches the ``charts``
        table, keeping load time in the single-digit millisecond range
        regardless of how many experiments have been saved.

        Returns
        -------
        pd.DataFrame
            Columns: experiment_id, model, balance_strategy, parameters,
            param_hash, baseline_accuracy, baseline_precision_macro,
            baseline_recall_macro, baseline_f1_macro, baseline_roc_auc_macro,
            enhanced_accuracy, …
        """
        sql = """
            SELECT
                e.id               AS experiment_id,
                e.model,
                e.balance_strategy,
                e.parameters,
                e.param_hash,
                b.accuracy         AS baseline_accuracy,
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

        This is intentionally separated from :meth:`load_experiments_summary`
        so the Dash dashboard never pays the cost of deserializing large roc /
        pr / confusion JSON unless the user actually opens the model modal.

        Parameters
        ----------
        experiment_id : int
            Primary key from the ``experiments`` table.
        phase : 'baseline' or 'enhanced'
        chart_type : str or None
            One of ``'roc_curve'``, ``'pr_curve'``, ``'confusion_matrix'``.
            ``None`` fetches all three.

        Returns
        -------
        dict
            ``{chart_type: deserialized_data}`` for each chart found.
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

        Returns
        -------
        dict
            Keys: ``feature_method`` (str), ``feature_parameters`` (dict|None),
            ``features`` (list), ``weights`` (list|dict|None).
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

        Chart placeholders (``None``) are injected for every chart key so
        that dashboard code can check ``if model["baseline"]["evaluation"]["charts"]["roc_curve"]``
        without a ``KeyError``.  Actual chart data is fetched lazily via
        :meth:`hydrate_model_charts` when the user opens a model-detail modal.

        Returns
        -------
        dict
            Keyed by ``"{slug}__{strategy}__{param_hash}"``.  Each value is::

                {
                    "experiment_id":    int,
                    "model":            str,
                    "balance_strategy": str,
                    "parameters":       dict,
                    "baseline": {
                        "evaluation": {
                            "metrics": {accuracy, precision_macro, …},
                            "charts":  {roc_curve: None, pr_curve: None, confusion_matrix: None},
                        },
                        "features":       [],
                        "feature_method": "",
                    },
                    "enhanced": { … },
                }
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

        Call this only when the user opens the model-detail modal — never
        during the initial leaderboard load.

        Parameters
        ----------
        model_entry : dict
            A single value from the dict returned by :meth:`load_models_dict`.
            **Mutated in-place.**

        Returns
        -------
        dict
            The same ``model_entry`` object with charts and features populated.

        Example
        -------
        ::

            models = repo.load_models_dict()
            repo.hydrate_model_charts(models[selected_key])
            roc = models[selected_key]["baseline"]["evaluation"]["charts"]["roc_curve"]
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

        This method exists purely for backward compatibility with callers that
        expect a DataFrame with ``baseline_confusion_matrix``,
        ``baseline_roc_curve``, etc.  It is **significantly slower** than
        :meth:`load_experiments_summary` because it fetches chart blobs.

        Prefer :meth:`load_experiments_summary` + :meth:`load_charts` for new
        code so that chart data is only fetched when actually needed.

        Returns
        -------
        pd.DataFrame  – one row per experiment, all columns present.
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
