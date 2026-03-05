import os
import json
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


def save_training_results(results_df, class_mappings_dict, base_output_folder, model_names, start_time, end_time, total_run_time):
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

    Returns
    -------
    dict
        Metadata dictionary containing summary information about the experiment.
    str
        Path to the run folder created.
    """
    os.makedirs(base_output_folder, exist_ok=True)

    date_str = datetime.now().strftime("%Y_%m_%d")
    run_index = 1
    while True:
        run_folder_name = f"{date_str}_{run_index:02d}"
        run_folder_path = os.path.join(base_output_folder, run_folder_name)
        if not os.path.exists(run_folder_path):
            os.makedirs(run_folder_path)
            break
        run_index += 1

    # Save results as parquet — preserves Python objects (lists, dicts) better than CSV
    results_path = os.path.join(run_folder_path, "results.parquet")
    results_df.to_parquet(results_path, index=False)

    # Save class mappings JSON
    class_mappings_path = os.path.join(run_folder_path, "class_mappings.json")
    with open(class_mappings_path, "w") as f:
        json.dump(_sanitize_for_json(class_mappings_dict), f, indent=4)

    # Compute metadata
    metadata = {
        "timestamp":                datetime.now().isoformat(),
        "num_models_trained":       len(model_names),
        "model_names":              model_names,
        "num_experiments":          len(results_df),
        "training_start_time":      start_time,
        "training_end_time":        end_time,
        "training_run_time":        total_run_time,
        "results_file_path":        results_path,
        "class_mappings_file_path": class_mappings_path,
    }

    # Save metadata JSON
    metadata_path = os.path.join(run_folder_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Results, class mappings, and metadata saved to {run_folder_path}")
    return metadata, run_folder_path