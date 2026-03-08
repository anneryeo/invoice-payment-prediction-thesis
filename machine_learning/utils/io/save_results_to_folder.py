import os
import json
import pickle
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime


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
        name = re.sub(r"[^\w]", "_", name)          # replace unsafe chars
        if name[0].isdigit():                         # identifiers can't start with a digit
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
        # ── results table ────────────────────────────────────────────────────
        _prepare_df_for_sqlite(model_results_df).to_sql(
            "results", con, if_exists="replace", index=False
        )

        # ── class_mappings table ─────────────────────────────────────────────
        con.execute(
            "CREATE TABLE IF NOT EXISTS class_mappings (id INTEGER PRIMARY KEY, data TEXT)"
        )
        con.execute("DELETE FROM class_mappings")
        con.execute(
            "INSERT INTO class_mappings (id, data) VALUES (1, ?)",
            (json.dumps(_sanitize_for_json(class_mappings_dict)),),
        )

        # ── survival_results table ───────────────────────────────────────────
        con.execute(
            "CREATE TABLE IF NOT EXISTS survival_results (id INTEGER PRIMARY KEY, data TEXT)"
        )
        con.execute("DELETE FROM survival_results")
        con.execute(
            "INSERT INTO survival_results (id, data) VALUES (1, ?)",
            (json.dumps(_sanitize_for_json(survival_results_dict)),),
        )

        # ── metadata table ───────────────────────────────────────────────────
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


def _load_sqlite(run_folder_path):
    """
    Load results, class_mappings, survival_results, and metadata from results.db.

    Columns that contain JSON-serialized dicts or lists are automatically
    deserialized back to native Python objects.

    Returns
    -------
    pd.DataFrame, dict, dict, dict
    """
    db_path = os.path.join(run_folder_path, "results.db")
    con = sqlite3.connect(db_path)
    try:
        model_results_df = pd.read_sql("SELECT * FROM results", con)

        # Deserialize any JSON-encoded object columns
        for col in model_results_df.columns:
            if model_results_df[col].dtype == object:
                model_results_df[col] = model_results_df[col].apply(_try_json_loads)

        row = con.execute("SELECT data FROM class_mappings WHERE id=1").fetchone()
        class_mappings = json.loads(row[0]) if row else {}

        # survival_results table may not exist in older databases
        try:
            row = con.execute("SELECT data FROM survival_results WHERE id=1").fetchone()
            survival_results = json.loads(row[0]) if row else {}
        except sqlite3.OperationalError:
            survival_results = {}

        row = con.execute("SELECT data FROM metadata WHERE id=1").fetchone()
        metadata = json.loads(row[0]) if row else {}
    finally:
        con.close()

    return model_results_df, class_mappings, survival_results, metadata


def _try_json_loads(value):
    """Attempt to deserialize a JSON string; return the original value on failure."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def save_training_results(model_results_df, survival_results_dict, class_mappings_dict, base_output_folder,
                          model_names, start_time, end_time, total_run_time,
                          format="sqlite"):
    """
    Save training results, survival results, class mappings, and metadata to
    a dynamically created run folder.

    Parameters
    ----------
    model_results_df : pd.DataFrame
        Flat DataFrame where each row is one experiment run. Contains all
        baseline and enhanced evaluation metrics, curve data, and feature
        selection info as columns.
    survival_results_dict : dict
        Dictionary containing survival analysis results, e.g.::

            {
                "best_c_index": <float>,
                "best_surv_parameters": <dict>,
            }

    class_mappings_dict : dict
        Dictionary mapping original class labels to their encoded
        integer representations.
    base_output_folder : str
        Base path where run folders will be created.
    model_names : list of str
        List of model names that were trained in this experiment run.
    start_time : str
        ISO format timestamp when training started.
    end_time : str
        ISO format timestamp when training ended.
    total_run_time : str
        Total run time as a formatted string.
    format : {"pickle", "excel", "sqlite"}, default "sqlite"
        Storage format for the results DataFrame.

        - ``"pickle"`` – saves ``results.pkl``, ``class_mappings.json``,
          ``survival_results.json``, and ``metadata.json``. Preserves all
          Python objects exactly. Recommended for further programmatic use.
        - ``"excel"``  – saves ``results.xlsx``, ``class_mappings.json``,
          ``survival_results.json``, and ``metadata.json``. Dict/list columns
          are serialized to JSON strings for compatibility. Recommended for
          manual inspection or sharing.
        - ``"sqlite"`` – saves a single ``results.db`` containing four tables:
          ``results``, ``class_mappings``, ``survival_results``, and
          ``metadata``. Dict/list columns are JSON-serialized and
          transparently deserialized on load. Recommended when you want SQL
          query access or a self-contained, language-agnostic file.

    Returns
    -------
    dict
        Metadata dictionary containing summary information about the experiment.
    str
        Path to the run folder created.

    Raises
    ------
    ValueError
        If format is not one of "pickle", "excel", or "sqlite".
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

    # Build metadata first so it can be embedded in the SQLite db too
    metadata = {
        "timestamp":                datetime.now().isoformat(),
        "num_models_trained":       len(model_names),
        "model_names":              model_names,
        "num_experiments":          len(model_results_df),
        "results_format":           format,
        "training_start_time":      start_time,
        "training_end_time":        end_time,
        "training_run_time":        total_run_time,
        "run_folder_path":          run_folder_path,
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
            model_results_df, class_mappings_dict, survival_results_dict, metadata, run_folder_path
        )
        metadata["results_file_path"] = db_path

    print(f"Results saved as {format} to {run_folder_path}")
    return metadata, run_folder_path


def load_training_results(run_folder_path):
    """
    Load training results, class mappings, survival results, and metadata
    from a run folder. Automatically detects whether results were saved as
    pickle, Excel, or SQLite based on the format recorded in metadata.

    Parameters
    ----------
    run_folder_path : str
        Path to the run folder created by save_training_results.

    Returns
    -------
    pd.DataFrame
        Results DataFrame.
    dict
        Class mappings dictionary.
    dict
        Survival results dictionary (keys: ``best_c_index``,
        ``best_surv_parameters``). Empty dict for runs saved before this
        field was introduced.
    dict
        Metadata dictionary.

    Notes
    -----
    Excel-loaded DataFrames will have dict/list columns stored as JSON strings.
    Use json.loads() to deserialize individual cells if needed.

    SQLite-loaded DataFrames have JSON columns deserialized automatically.
    """
    # SQLite stores metadata inside the db; detect it by file presence
    db_path = os.path.join(run_folder_path, "results.db")
    if os.path.exists(db_path) and not os.path.exists(
        os.path.join(run_folder_path, "metadata.json")
    ):
        return _load_sqlite(run_folder_path)

    with open(os.path.join(run_folder_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    fmt = metadata.get("results_format", "pickle")

    if fmt == "sqlite":
        return _load_sqlite(run_folder_path)
    elif fmt == "excel":
        model_results_df = pd.read_excel(os.path.join(run_folder_path, "results.xlsx"))
    else:
        with open(os.path.join(run_folder_path, "results.pkl"), "rb") as f:
            model_results_df = pickle.load(f)

    with open(os.path.join(run_folder_path, "class_mappings.json"), "r") as f:
        class_mappings = json.load(f)

    # survival_results.json may not exist in older run folders
    survival_results_path = os.path.join(run_folder_path, "survival_results.json")
    if os.path.exists(survival_results_path):
        with open(survival_results_path, "r") as f:
            survival_results = json.load(f)
    else:
        survival_results = {}

    return model_results_df, class_mappings, survival_results, metadata