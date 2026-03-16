import os
import json
import pickle
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
#  SERIALIZATION HELPERS  (save-side only)
# ══════════════════════════════════════════════════════════════════════════════

def _sanitize_for_json(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
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


def _prepare_df_for_excel(df):
    """
    Coerce object columns containing dicts or lists to JSON strings so that
    Excel can store them without raising serialization errors.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: json.dumps(_sanitize_for_json(x))
                if isinstance(x, (dict, list)) else x
            )
    return df


def _sanitize_column_names(df):
    """
    Ensure all DataFrame column names are non-empty strings safe for use as
    SQLite identifiers.  Blank/whitespace-only names are replaced with
    ``col_<position>``, and any character that is not alphanumeric or an
    underscore is replaced with ``_``.
    """
    import re
    new_cols = []
    for i, col in enumerate(df.columns):
        name = str(col).strip()
        if not name:
            name = f"col_{i}"
        name = re.sub(r"[^\w]", "_", name)
        if name[0].isdigit():
            name = f"col_{name}"
        new_cols.append(name)
    df.columns = new_cols
    return df


def _prepare_df_for_sqlite(df):
    """
    Coerce object columns containing dicts, lists, or numpy types to JSON
    strings so that SQLite can store them.  Numeric numpy scalars in
    otherwise-numeric columns are also cast to native Python floats/ints.
    Column names are sanitized to be valid SQLite identifiers.
    """
    df = df.copy()
    df = _sanitize_column_names(df)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: json.dumps(_sanitize_for_json(x))
                if isinstance(x, (dict, list, np.ndarray)) else x
            )
        elif np.issubdtype(df[col].dtype, np.integer):
            df[col] = df[col].astype(int)
        elif np.issubdtype(df[col].dtype, np.floating):
            df[col] = df[col].astype(float)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SQLITE WRITER
# ══════════════════════════════════════════════════════════════════════════════

def _save_sqlite(model_results_df, class_mappings_dict, survival_results_dict, metadata, run_folder_path):
    """
    Write all artefacts into a single SQLite database file
    (results.db) inside *run_folder_path*.

    Tables
    ------
    results           – one row per experiment, columns mirror the DataFrame.
    class_mappings    – single row: id=1, data TEXT (JSON blob).
    survival_results  – single row: id=1, data TEXT (JSON blob).
    metadata          – single row: id=1, data TEXT (JSON blob).
    """
    db_path = os.path.join(run_folder_path, "results.db")
    con = sqlite3.connect(db_path)
    try:
        _prepare_df_for_sqlite(model_results_df).to_sql(
            "results", con, if_exists="replace", index=False
        )

        con.execute(
            "CREATE TABLE IF NOT EXISTS class_mappings (id INTEGER PRIMARY KEY, data TEXT)"
        )
        con.execute("DELETE FROM class_mappings")
        con.execute(
            "INSERT INTO class_mappings (id, data) VALUES (1, ?)",
            (json.dumps(_sanitize_for_json(class_mappings_dict)),),
        )

        con.execute(
            "CREATE TABLE IF NOT EXISTS survival_results (id INTEGER PRIMARY KEY, data TEXT)"
        )
        con.execute("DELETE FROM survival_results")
        con.execute(
            "INSERT INTO survival_results (id, data) VALUES (1, ?)",
            (json.dumps(_sanitize_for_json(survival_results_dict)),),
        )

        con.execute(
            "CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, data TEXT)"
        )
        con.execute("DELETE FROM metadata")
        con.execute(
            "INSERT INTO metadata (id, data) VALUES (1, ?)",
            (json.dumps(metadata),),
        )

        con.commit()
    finally:
        con.close()

    return db_path


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def save_training_results(
    model_results_df, survival_results_dict, class_mappings_dict,
    base_output_folder, model_names, start_time, end_time, total_run_time,
    format="sqlite",
):
    """
    Save training results, survival results, class mappings, and metadata to
    a dynamically created run folder.

    Parameters
    ----------
    model_results_df : pd.DataFrame
        Flat DataFrame where each row is one experiment run.
    survival_results_dict : dict
        Survival analysis results (best_c_index, best_surv_parameters, …).
    class_mappings_dict : dict
        Original class labels → encoded integer representations.
    base_output_folder : str
        Base path where run folders will be created.
    model_names : list of str
        Names of models trained in this run.
    start_time : str
        ISO-format timestamp when training started.
    end_time : str
        ISO-format timestamp when training ended.
    total_run_time : str
        Total run time as a formatted string.
    format : {"pickle", "excel", "sqlite"}, default "sqlite"
        Storage format.

    Returns
    -------
    metadata : dict
    run_folder_path : str
    """
    if format not in ("pickle", "excel", "sqlite"):
        raise ValueError(f"Invalid format {format!r}. Must be 'pickle', 'excel', or 'sqlite'.")

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

    if format == "pickle":
        results_path = os.path.join(run_folder_path, "results.pkl")
        with open(results_path, "wb") as f:
            pickle.dump(model_results_df, f)
        metadata["results_file_path"] = results_path

        class_mappings_path = os.path.join(run_folder_path, "class_mappings.json")
        with open(class_mappings_path, "w") as f:
            json.dump(_sanitize_for_json(class_mappings_dict), f, indent=4)
        metadata["class_mappings_file_path"] = class_mappings_path

        survival_results_path = os.path.join(run_folder_path, "survival_results.json")
        with open(survival_results_path, "w") as f:
            json.dump(_sanitize_for_json(survival_results_dict), f, indent=4)
        metadata["survival_results_file_path"] = survival_results_path

        metadata_path = os.path.join(run_folder_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    elif format == "excel":
        results_path = os.path.join(run_folder_path, "results.xlsx")
        _prepare_df_for_excel(model_results_df).to_excel(results_path, index=False)
        metadata["results_file_path"] = results_path

        class_mappings_path = os.path.join(run_folder_path, "class_mappings.json")
        with open(class_mappings_path, "w") as f:
            json.dump(_sanitize_for_json(class_mappings_dict), f, indent=4)
        metadata["class_mappings_file_path"] = class_mappings_path

        survival_results_path = os.path.join(run_folder_path, "survival_results.json")
        with open(survival_results_path, "w") as f:
            json.dump(_sanitize_for_json(survival_results_dict), f, indent=4)
        metadata["survival_results_file_path"] = survival_results_path

        metadata_path = os.path.join(run_folder_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    else:  # sqlite
        db_path = _save_sqlite(
            model_results_df, class_mappings_dict, survival_results_dict,
            metadata, run_folder_path,
        )
        metadata["results_file_path"] = db_path

    print(f"Results saved as {format} to {run_folder_path}")
    return metadata, run_folder_path