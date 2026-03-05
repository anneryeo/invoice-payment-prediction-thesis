import os
import json
import numpy as np
from datetime import datetime

class _NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts numpy scalar types to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_training_results(results_dict, class_mappings_dict, base_output_folder, model_names, start_time, end_time, total_run_time):
    """
    Save training results, class mappings, and metadata to a dynamically created run folder.

    Parameters
    ----------
    results_dict : dict
        Dictionary keyed by model name containing evaluation results
        (metrics + raw chart data) and features.
    class_mappings_dict : dict
        Dictionary mapping original class labels to their encoded
        integer representations.
    base_output_folder : str
        Base path where run folders will be created.
    model_names : list of str
        List of model names that were trained in this experiment run.
    start_time : float
        Timestamp (from time.time()) when training started.
    end_time : float
        Timestamp (from time.time()) when training ended.
    total_run_time : float
        Total run time in seconds.

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

    # Save results JSON
    results_path = os.path.join(run_folder_path, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=4, cls=_NumpyEncoder)

    # Save class mappings JSON
    class_mappings_path = os.path.join(run_folder_path, "class_mappings.json")
    with open(class_mappings_path, "w") as f:
        json.dump(class_mappings_dict, f, indent=4, cls=_NumpyEncoder)

    # Compute metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "num_models_trained": len(model_names),
        "model_names": model_names,
        "training_start_time": start_time,
        "training_end_time": end_time,
        "training_run_time": total_run_time,
        "results_file_path": results_path,
        "class_mappings_file_path": class_mappings_path
    }

    # Save metadata JSON
    metadata_path = os.path.join(run_folder_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4, cls=_NumpyEncoder)

    print(f"Results, class mappings, and metadata saved to {run_folder_path}")
    return metadata, run_folder_path