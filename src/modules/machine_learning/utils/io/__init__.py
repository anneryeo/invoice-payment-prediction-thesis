# machine_learning/utils/io/__init__.py
#
# Public API for the I/O subsystem.
#
# This package is organized into four concerns:
#
#   Schema & storage
#   ────────────────
#   db_schema              – DDL strings and SCHEMA_VERSION
#   results_repository     – ResultsRepository (OOP SQLite interface)
#   save_results_to_folder – save_training_results() writer
#   migrate_db_schema      – Schema migrators (V1→V2, V3→V4)
#
#   Reading & session management
#   ────────────────────────────
#   load_results_from_folder – load_training_results(), SessionStore
#   data_loaders             – Dashboard helpers (load_models_from_results)
#
#   Analysis (NEW)
#   ──────────────
#   analysis                 – ResultsAnalyzer fluent API
#   analysis.registry        – Model/strategy metadata constants
#   analysis.visualization   – Theme, plot functions
#
# Import shortcuts:
#
#   from machine_learning.utils.io import SessionStore, ResultsRepository
#   from machine_learning.utils.io import ResultsAnalyzer
#   from machine_learning.utils.io import save_training_results, load_training_results

# ── Schema & storage ─────────────────────────────────────────────────────────
from .db_schema import SCHEMA_VERSION
from .results_repository import ResultsRepository

# ── Reading & session management ─────────────────────────────────────────────
from .load_results_from_folder import SessionStore, load_training_results
from .save_results_to_folder import save_training_results
from .data_loaders import load_models_from_results, get_repository

# ── Analysis ─────────────────────────────────────────────────────────────────
from .analysis import ResultsAnalyzer

__all__ = [
    # Schema
    "SCHEMA_VERSION",
    "ResultsRepository",
    # Session I/O
    "SessionStore",
    "load_training_results",
    "save_training_results",
    # Dashboard helpers
    "load_models_from_results",
    "get_repository",
    # Analysis (new)
    "ResultsAnalyzer",
]
