import json
import os

def read_settings_json(file_path="settings.json"):
    """
    Reads a JSON file and returns its contents as a Python dictionary.
    Strips out any lines starting with '//' to allow comments.

    Parameters:
        file_path (str): Path to the JSON file. Defaults to Thesis/settings.json.

    Returns:
        dict: Parsed JSON data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, "r", encoding="utf-8") as f:
        # Remove comment lines starting with //
        content = "\n".join(
            line for line in f if not line.strip().startswith("//")
        )
        data = json.loads(content)
    
    return data