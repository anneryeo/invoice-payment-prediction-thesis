import os
import json
import subprocess
from pathlib import Path

def setup_dev_env():
    print("🚀 Initializing Private Developer Environment...")
    
    # 1. Get current project root
    root = Path(__file__).resolve().parent
    print(f"📍 Project Root Detected: {root}")

    # 2. Privacy Guard: Ensure SENSITIVE files are ignored, but SHARED files are kept
    gitignore_path = root / ".gitignore"
    # We ignore the local config and cache, but NOT this setup script
    ignored_items = ["settings.json", ".env", "data/cache/", "__pycache__/", "*.pyc", "convert_notebooks.py"]
    
    if gitignore_path.exists():
        with open(gitignore_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        with open(gitignore_path, "a", encoding="utf-8") as f:
            for item in ignored_items:
                if item not in content:
                    f.write(f"\n{item}")
                    print(f"🔒 Added {item} to .gitignore")
    else:
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ignored_items))
        print("🆕 Created .gitignore with privacy rules")

    # 3. Metadata Scrubber: Automatic Notebook Output Stripping
    print("Sweep: Setting up Automatic Metadata Scrubber for Git...")
    if (root / ".git").exists():
        try:
            # Configures git to use nbstripout (runs via npx to avoid dependency hell)
            subprocess.run(["git", "config", "filter.nbstripout.clean", "npx -y nbstripout"], check=True)
            subprocess.run(["git", "config", "filter.nbstripout.smudge", "cat"], check=True)
            
            attributes_path = root / ".gitattributes"
            attr_content = "*.ipynb filter=nbstripout\n"
            
            if not attributes_path.exists() or attr_content not in open(attributes_path, encoding="utf-8").read():
                with open(attributes_path, "a", encoding="utf-8") as f:
                    f.write(attr_content)
                print("✅ Git will now automatically strip outputs/usernames from Notebooks before every commit.")
        except Exception as e:
            print(f"⚠️ Git filter setup failed: {e}. (This requires Node.js for 'npx')")
    else:
        print("⚠️ No .git folder found. Skipping Git filter setup.")

    # 4. Generate Local settings.json
    settings_path = root / "settings.json"
    default_config = {
        "Config": {
            "debug_mode": "False",
            "TEMP_CACHE": "data/temp_cache"
        },
        "TrainingInput": {
            "CHART_OF_ACCOUNTS": str(root / "database" / "chart_of_accounts.xlsx"),
            "ENROLLEES": str(root / "database" / "enrollees_pseudonymized.xlsx"),
            "REVENUES": str(root / "database" / "revenues_pseudonymized.xlsx")
        },
        "Training": {
            "MODEL_PARAMETERS": "src/modules/machine_learning/parameters.json",
            "RESULTS_ROOT": "results",
            "LOGS": "data/logs",
            "DEPLOYED_MODELS": "results/deployed_models",
            "observation_end": "2026/04/24",
            "target_feature": "dtp_bracket",
            "test_size": "0.30"
        }
    }

    if not settings_path.exists():
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)
        print("✅ Generated local settings.json with your machine's absolute paths.")
    else:
        print("ℹ️ settings.json already exists.")

    # 5. Cell Snippet for Notebooks
    snippet = """
# --- PASTE THIS IN YOUR FIRST NOTEBOOK CELL ---
from pathlib import Path
import sys

# Automatically find repo root by looking for .git
ROOT = Path.cwd()
while not (ROOT / ".git").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.data_loaders.read_settings_json import read_settings_json
settings = read_settings_json(ROOT / "settings.json")
# ----------------------------------------------
"""
    print("\n💡 Share this snippet with your co-devs for their Notebook cells:")
    print(snippet)

if __name__ == "__main__":
    setup_dev_env()
