import json
import os

def read_settings_json(file_path="settings.json"):
    """
    Reads a JSON file and returns its contents as a Python dictionary.
    Strips out any comments starting with '//' (full-line or inline).

    Parameters:
        file_path (str): Path to the JSON file. Defaults to settings.json.

    Returns:
        dict: Parsed JSON data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, "r", encoding="utf-8") as f:
        cleaned_lines = []
        for line in f:
            # Remove inline comments
            if "//" in line:
                line = line.split("//", 1)[0]
            if line.strip():  # keep non-empty lines
                cleaned_lines.append(line)
        
        content = "\n".join(cleaned_lines)
    
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON after stripping comments: {e}")
    
    return data