# machine_learning/utils/io/db_schema.py
#
# Single source of truth for the normalized SQLite schema used by
# ResultsRepository.  Import the DDL strings and SCHEMA_VERSION here;
# never hard-code table names or column lists elsewhere.
#
# Schema overview
# ───────────────
#   experiments      – one row per model run (lightweight identifiers only)
#   metrics          – scalar evaluation metrics, one row per experiment × phase
#   charts           – heavy JSON blobs (roc_curve / pr_curve / confusion_matrix),
#                      one row per experiment × phase × chart_type; fetched on demand
#   features         – feature selection results, one row per experiment × phase
#   class_mappings   – single-row JSON blob (label → int encoding)
#   survival_results – single-row JSON blob (best_c_index, best_surv_parameters, …)
#   metadata         – single-row JSON blob (run timestamps, model list, …)
#   schema_version   – single-row version sentinel for future migrations


# ── Version ───────────────────────────────────────────────────────────────────

SCHEMA_VERSION: int = 4
"""
Increment this whenever a breaking DDL change is introduced so that
ResultsRepository can detect and migrate older databases automatically.

v4 — added cache_key column to experiments and created cache_registry table.
v3 — added undersample_threshold column to experiments.
"""


# ── Core experiment tables ────────────────────────────────────────────────────

DDL_EXPERIMENTS = """
CREATE TABLE IF NOT EXISTS experiments (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    model                TEXT    NOT NULL,
    balance_strategy     TEXT    NOT NULL DEFAULT 'none',
    undersample_threshold REAL,
    parameters           TEXT,
    param_hash           TEXT,
    cache_key            TEXT,
    created_at           TEXT    DEFAULT (datetime('now'))
)
"""

# ── Caching registry ─────────────────────────────────────────────────────────

DDL_CACHE_REGISTRY = """
CREATE TABLE IF NOT EXISTS cache_registry (
    cache_key       TEXT PRIMARY KEY,
    cache_type      TEXT NOT NULL CHECK(cache_type IN ('dataset', 'model')),
    parameters_hash TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    created_at      TEXT DEFAULT (datetime('now')),
    metadata        TEXT
)
"""

DDL_METRICS = """
CREATE TABLE IF NOT EXISTS metrics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id    INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    phase            TEXT    NOT NULL CHECK(phase IN ('baseline', 'enhanced')),
    accuracy         REAL,
    precision_macro  REAL,
    recall_macro     REAL,
    f1_macro         REAL,
    roc_auc_macro    REAL
)
"""

DDL_CHARTS = """
CREATE TABLE IF NOT EXISTS charts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id    INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    phase            TEXT    NOT NULL CHECK(phase IN ('baseline', 'enhanced')),
    chart_type       TEXT    NOT NULL
                             CHECK(chart_type IN ('roc_curve', 'pr_curve', 'confusion_matrix')),
    data             TEXT
)
"""

DDL_FEATURES = """
CREATE TABLE IF NOT EXISTS features (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id       INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    phase               TEXT    NOT NULL CHECK(phase IN ('baseline', 'enhanced')),
    feature_method      TEXT,
    feature_parameters  TEXT,
    features_json       TEXT,
    weights_json        TEXT
)
"""


# ── Blob / singleton tables ───────────────────────────────────────────────────

DDL_CLASS_MAPPINGS = """
CREATE TABLE IF NOT EXISTS class_mappings (
    id   INTEGER PRIMARY KEY,
    data TEXT
)
"""

DDL_SURVIVAL_RESULTS = """
CREATE TABLE IF NOT EXISTS survival_results (
    id   INTEGER PRIMARY KEY,
    data TEXT
)
"""

DDL_METADATA = """
CREATE TABLE IF NOT EXISTS metadata (
    id   INTEGER PRIMARY KEY,
    data TEXT
)
"""

DDL_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    id      INTEGER PRIMARY KEY,
    version INTEGER NOT NULL
)
"""


# ── Indexes (applied after table creation) ────────────────────────────────────

DDL_INDEXES = [
    # Leaderboard query joins metrics twice (baseline + enhanced); covering
    # indexes on (experiment_id, phase) keep this a pure index scan.
    "CREATE INDEX IF NOT EXISTS idx_metrics_exp_phase  ON metrics  (experiment_id, phase)",
    "CREATE INDEX IF NOT EXISTS idx_charts_exp_phase   ON charts   (experiment_id, phase)",
    "CREATE INDEX IF NOT EXISTS idx_features_exp_phase ON features (experiment_id, phase)",
]


# ── Ordered list consumed by ResultsRepository.initialize_schema() ────────────

ALL_DDL: list[str] = [
    DDL_EXPERIMENTS,
    DDL_CACHE_REGISTRY,
    DDL_METRICS,
    DDL_CHARTS,
    DDL_FEATURES,
    DDL_CLASS_MAPPINGS,
    DDL_SURVIVAL_RESULTS,
    DDL_METADATA,
    DDL_SCHEMA_VERSION,
    *DDL_INDEXES,
]
