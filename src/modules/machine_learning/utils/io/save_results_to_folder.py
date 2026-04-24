# machine_learning/utils/io/save_results_to_folder.py
#
# Public API for persisting a training session to disk.
#
# Storage layout
# ──────────────
# Each call to save_training_results() creates a dated run folder:
#
#     <base_output_folder>/
#       YYYY_MM_DD_01/
#         results.db          ← SQLite v2 normalized schema (default)
#       YYYY_MM_DD_02/        ← auto-incremented when the date folder exists
#
# The SQLite writer delegates all schema knowledge and I/O to
# ResultsRepository so that this file stays thin and focused on the
# folder-naming / format-routing concern only.
#
# Legacy formats (pickle / excel) are preserved for backward compatibility
# but are no longer the default.

import json
import os
import pickle
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from .results_repository import ResultsRepository


# ══════════════════════════════════════════════════════════════════════════════
#  SERIALIZATION HELPERS  (pickle / excel paths only)
# ══════════════════════════════════════════════════════════════════════════════

def _sanitize_for_json(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _prepare_df_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce object columns that contain dicts or lists to JSON strings so that
    openpyxl can write them without raising serialization errors.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: json.dumps(_sanitize_for_json(x))
                if isinstance(x, (dict, list)) else x
            )
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  LEGACY FORMAT WRITERS  (pickle / excel)
# ══════════════════════════════════════════════════════════════════════════════

def _save_pickle(
    model_results_df: pd.DataFrame,
    class_mappings_dict: dict,
    survival_results_dict: dict,
    metadata: dict,
    run_folder_path: str,
) -> str:
    results_path = os.path.join(run_folder_path, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(model_results_df, f)

    with open(os.path.join(run_folder_path, "class_mappings.json"), "w") as f:
        json.dump(_sanitize_for_json(class_mappings_dict), f, indent=4)

    with open(os.path.join(run_folder_path, "survival_results.json"), "w") as f:
        json.dump(_sanitize_for_json(survival_results_dict), f, indent=4)

    metadata_path = os.path.join(run_folder_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return results_path


def _save_excel(
    model_results_df: pd.DataFrame,
    class_mappings_dict: dict,
    survival_results_dict: dict,
    metadata: dict,
    run_folder_path: str,
) -> str:
    results_path = os.path.join(run_folder_path, "results.xlsx")
    _prepare_df_for_excel(model_results_df).to_excel(results_path, index=False)

    with open(os.path.join(run_folder_path, "class_mappings.json"), "w") as f:
        json.dump(_sanitize_for_json(class_mappings_dict), f, indent=4)

    with open(os.path.join(run_folder_path, "survival_results.json"), "w") as f:
        json.dump(_sanitize_for_json(survival_results_dict), f, indent=4)

    metadata_path = os.path.join(run_folder_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return results_path


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def save_training_results(
    model_results_df: pd.DataFrame,
    survival_results_dict: dict,
    class_mappings_dict: Optional[dict],
    base_output_folder: str,
    model_names: list,
    start_time: str,
    end_time: str,
    total_run_time: str,
    format: str = "sqlite",
) -> tuple[dict, str]:
    """
    Save training results, survival results, class mappings, and metadata to
    a dynamically created, date-stamped run folder.

    The SQLite format (default) writes to a fully normalized v2 schema via
    :class:`ResultsRepository`, separating heavy chart blobs from scalar
    metrics so the dashboard leaderboard loads instantly on future sessions.

    Parameters
    ----------
    model_results_df : pd.DataFrame
        Flat results DataFrame produced by SurvivalExperimentRunner.run().
    survival_results_dict : dict
        Survival analysis results (best_c_index, best_surv_parameters, …).
    class_mappings_dict : dict
        Original class labels → encoded integer representations.
    base_output_folder : str
        Root path under which dated run folders are created.
    model_names : list of str
        Names of models included in this training run.
    start_time : str
        ISO-format timestamp when training started.
    end_time : str
        ISO-format timestamp when training ended.
    total_run_time : str
        Human-readable total run duration.
    format : {'sqlite', 'pickle', 'excel'}, default 'sqlite'
        Storage format.  ``'sqlite'`` is strongly recommended for all new
        sessions; ``'pickle'`` and ``'excel'`` are kept for compatibility.

    Returns
    -------
    metadata : dict
    run_folder_path : str
    """
    if format not in ("pickle", "excel", "sqlite"):
        raise ValueError(
            f"Invalid format {format!r}. Choose 'sqlite', 'pickle', or 'excel'."
        )

    # ── Create the dated run folder ───────────────────────────────────────────
    os.makedirs(base_output_folder, exist_ok=True)

    date_str  = datetime.now().strftime("%Y_%m_%d")
    run_index = 1
    while True:
        run_folder_name = f"{date_str}_{run_index:02d}"
        run_folder_path = os.path.join(base_output_folder, run_folder_name)
        if not os.path.exists(run_folder_path):
            os.makedirs(run_folder_path)
            break
        run_index += 1

    # ── Build metadata ────────────────────────────────────────────────────────
    metadata = {
        "timestamp":           datetime.now().isoformat(),
        "num_models_trained":  len(model_names),
        "model_names":         model_names,
        "num_experiments":     len(model_results_df),
        "results_format":      format,
        "training_start_time": start_time,
        "training_end_time":   end_time,
        "training_run_time":   total_run_time,
        "run_folder_path":     run_folder_path,
    }

    # ── Write ─────────────────────────────────────────────────────────────────
    if format == "sqlite":
        db_path = os.path.join(run_folder_path, "results.db")
        repo = ResultsRepository(db_path)
        repo.save_session(
            model_results_df,
            class_mappings_dict,
            survival_results_dict,
            metadata,
        )
        metadata["results_file_path"] = db_path

    elif format == "pickle":
        results_path = _save_pickle(
            model_results_df, class_mappings_dict,
            survival_results_dict, metadata, run_folder_path,
        )
        metadata["results_file_path"] = results_path

    else:  # excel
        results_path = _save_excel(
            model_results_df, class_mappings_dict,
            survival_results_dict, metadata, run_folder_path,
        )
        metadata["results_file_path"] = results_path

    print(f"Results saved as {format} to {run_folder_path}")
    return metadata, run_folder_path
