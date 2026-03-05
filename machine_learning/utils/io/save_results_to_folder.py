import os
import json
import pickle
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


def save_training_results(results_df, class_mappings_dict, base_output_folder,
                          model_names, start_time, end_time, total_run_time,
                          format="pickle"):
    """
    Save training results, class mappings, and metadata to a dynamically created run folder.

    Parameters
    ----------
    results_df : pd.DataFrame
        Flat DataFrame where each row is one experiment run. Contains all
        baseline and enhanced evaluation metrics, curve data, and feature
        selection info as columns.
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
    format : {"pickle", "excel"}, default "pickle"
        Storage format for the results DataFrame.
        - "pickle" : saves as results.pkl. Preserves all Python objects
          (dicts, lists) exactly. Recommended for further programmatic use.
        - "excel"  : saves as results.xlsx. Columns containing dicts or
          lists are serialized to JSON strings for compatibility. Recommended
          for manual inspection or sharing.

    Returns
    -------
    dict
        Metadata dictionary containing summary information about the experiment.
    str
        Path to the run folder created.

    Raises
    ------
    ValueError
        If format is not one of "pickle" or "excel".
    """
    if format not in ("pickle", "excel"):
        raise ValueError(f"Invalid format {format!r}. Must be 'pickle' or 'excel'.")

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

    # Save results in requested format
    if format == "pickle":
        results_path = os.path.join(run_folder_path, "results.pkl")
        with open(results_path, "wb") as f:
            pickle.dump(results_df, f)
    else:
        results_path = os.path.join(run_folder_path, "results.xlsx")
        _prepare_df_for_excel(results_df).to_excel(results_path, index=False)

    # Save class mappings JSON
    class_mappings_path = os.path.join(run_folder_path, "class_mappings.json")
    with open(class_mappings_path, "w") as f:
        json.dump(_sanitize_for_json(class_mappings_dict), f, indent=4)

    # Save metadata JSON
    metadata = {
        "timestamp":                datetime.now().isoformat(),
        "num_models_trained":       len(model_names),
        "model_names":              model_names,
        "num_experiments":          len(results_df),
        "results_format":           format,
        "training_start_time":      start_time,
        "training_end_time":        end_time,
        "training_run_time":        total_run_time,
        "results_file_path":        results_path,
        "class_mappings_file_path": class_mappings_path,
    }
    metadata_path = os.path.join(run_folder_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Results saved as {format} to {run_folder_path}")
    return metadata, run_folder_path


def load_training_results(run_folder_path):
    """
    Load training results, class mappings, and metadata from a run folder.
    Automatically detects whether results were saved as pickle or Excel
    based on the format recorded in metadata.json.

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
        Metadata dictionary.

    Notes
    -----
    Excel-loaded DataFrames will have dict/list columns stored as JSON strings.
    Use json.loads() to deserialize individual cells if needed.
    """
    with open(os.path.join(run_folder_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    if metadata.get("results_format", "pickle") == "excel":
        results_df = pd.read_excel(os.path.join(run_folder_path, "results.xlsx"))
    else:
        with open(os.path.join(run_folder_path, "results.pkl"), "rb") as f:
            results_df = pickle.load(f)

    with open(os.path.join(run_folder_path, "class_mappings.json"), "r") as f:
        class_mappings = json.load(f)

    return results_df, class_mappings, metadata