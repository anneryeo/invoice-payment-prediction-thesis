import os
import json
import subprocess
from pathlib import Path


def setup_dev_env():
    print("🚀 Initializing Private Developer Environment...")
    print("=" * 60)

    # Track all changes for a final summary
    changes_made = []

    # 1. Get current project root
    root = Path(__file__).resolve().parent
    print(f"\n📍 Project Root Detected: {root}")

    # -------------------------------------------------------------------------
    # 2. Privacy Guard: Ensure SENSITIVE files are ignored
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("📄 STEP 1: .gitignore — Privacy Guard")
    print("-" * 60)

    gitignore_path = root / ".gitignore"
    ignored_items = [
        "settings.json",
        ".env",
        "data/cache/",
        "__pycache__/",
        "*.pyc",
        "convert_notebooks.py",
    ]

    if gitignore_path.exists():
        with open(gitignore_path, "r", encoding="utf-8") as f:
            existing_content = f.read()

        items_to_add = [item for item in ignored_items if item not in existing_content]

        if items_to_add:
            with open(gitignore_path, "a", encoding="utf-8") as f:
                for item in items_to_add:
                    f.write(f"\n{item}")

            print(f"   ✏️  EDITED: {gitignore_path}")
            print(f"   ➕ Appended {len(items_to_add)} new entries:")
            for item in items_to_add:
                print(f"        + {item}")
            already_present = [item for item in ignored_items if item in existing_content]
            if already_present:
                print(f"   ⏭️  Skipped {len(already_present)} entries (already present):")
                for item in already_present:
                    print(f"        ~ {item}")
            changes_made.append(f".gitignore — appended {len(items_to_add)} entries")
        else:
            print(f"   ⏭️  NO CHANGES: {gitignore_path}")
            print("   All required entries are already present.")
    else:
        with open(gitignore_path, "w", encoding="utf-8") as f:
            content = "\n".join(ignored_items)
            f.write(content)
        print(f"   🆕 CREATED: {gitignore_path}")
        print(f"   Wrote {len(ignored_items)} entries:")
        for item in ignored_items:
            print(f"        + {item}")
        changes_made.append(f".gitignore — created with {len(ignored_items)} entries")

    # -------------------------------------------------------------------------
    # 3. Metadata Scrubber: Automatic Notebook Output Stripping
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("🧹 STEP 2: Git Filter — Notebook Metadata Scrubber")
    print("-" * 60)
    print("   Purpose: Automatically strip cell outputs, execution")
    print("   counts, and usernames from .ipynb files before each commit.")
    print("   Tool: nbstripout (invoked via npx — no install needed).")

    if (root / ".git").exists():
        try:
            # --- Git config: filter.nbstripout.clean ---
            clean_cmd = "npx -y nbstripout"
            result_clean = subprocess.run(
                ["git", "config", "filter.nbstripout.clean", clean_cmd],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"\n   ✏️  SET git config: filter.nbstripout.clean = '{clean_cmd}'")
            changes_made.append("git config — set filter.nbstripout.clean")

            # --- Git config: filter.nbstripout.smudge ---
            smudge_cmd = "cat"
            result_smudge = subprocess.run(
                ["git", "config", "filter.nbstripout.smudge", smudge_cmd],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"   ✏️  SET git config: filter.nbstripout.smudge = '{smudge_cmd}'")
            changes_made.append("git config — set filter.nbstripout.smudge")

            # --- .gitattributes ---
            attributes_path = root / ".gitattributes"
            attr_line = "*.ipynb filter=nbstripout\n"

            if attributes_path.exists():
                existing_attrs = open(attributes_path, encoding="utf-8").read()
                if attr_line.strip() in existing_attrs:
                    print(f"\n   ⏭️  NO CHANGES: {attributes_path}")
                    print(f"   Filter rule already present: '{attr_line.strip()}'")
                else:
                    with open(attributes_path, "a", encoding="utf-8") as f:
                        f.write(attr_line)
                    print(f"\n   ✏️  EDITED: {attributes_path}")
                    print(f"   ➕ Appended: '{attr_line.strip()}'")
                    changes_made.append(f".gitattributes — appended filter rule")
            else:
                with open(attributes_path, "w", encoding="utf-8") as f:
                    f.write(attr_line)
                print(f"\n   🆕 CREATED: {attributes_path}")
                print(f"   Wrote: '{attr_line.strip()}'")
                changes_made.append(f".gitattributes — created with filter rule")

            print("\n   ✅ Git will now automatically strip outputs/usernames")
            print("   from .ipynb files before every commit.")

        except FileNotFoundError:
            print("\n   ❌ FAILED: 'git' command not found on PATH.")
            print("   Please install Git and try again.")
        except subprocess.CalledProcessError as e:
            print(f"\n   ❌ FAILED: Git config command returned an error.")
            print(f"   stderr: {e.stderr.strip() if e.stderr else '(none)'}")
            print("   This requires Node.js for 'npx'. Is Node installed?")
        except Exception as e:
            print(f"\n   ⚠️  UNEXPECTED ERROR: {e}")
    else:
        print("\n   ⏭️  SKIPPED: No .git folder found at project root.")
        print("   Initialize a git repo first ('git init') to enable this.")

    # -------------------------------------------------------------------------
    # 4. Generate Local settings.json
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("⚙️  STEP 3: settings.json — Local Configuration")
    print("-" * 60)

    settings_path = root / "settings.json"
    default_config = {
        "Config": {
            "debug_mode": "False",
            "TEMP_CACHE": "data/temp_cache",
        },
        "TrainingInput": {
            "CHART_OF_ACCOUNTS": str(root / "database" / "chart_of_accounts.xlsx"),
            "ENROLLEES": str(root / "database" / "enrollees_pseudonymized.xlsx"),
            "REVENUES": str(root / "database" / "revenues_pseudonymized.xlsx"),
        },
        "Training": {
            "MODEL_PARAMETERS": "src/modules/machine_learning/parameters.json",
            "RESULTS_ROOT": "results",
            "LOGS": "data/logs",
            "DEPLOYED_MODELS": "results/deployed_models",
            "observation_end": "2026/04/24",
            "target_feature": "dtp_bracket",
            "test_size": "0.30",
        },
    }

    if not settings_path.exists():
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)

        print(f"   🆕 CREATED: {settings_path}")
        print("   Contents written:")
        for section, values in default_config.items():
            print(f"     [{section}]")
            for key, val in values.items():
                print(f"       {key} = {val}")
        changes_made.append("settings.json — created with default config")
    else:
        print(f"   ⏭️  NO CHANGES: {settings_path}")
        print("   File already exists. Delete it manually to regenerate.")

    # -------------------------------------------------------------------------
    # 5. Cell Snippet for Notebooks
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("📋 STEP 4: Notebook Boilerplate Snippet (info only)")
    print("-" * 60)
    print("   No files modified. Copy-paste this into your first cell:\n")

    snippet = """\
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
# -----------------------------------------------"""
    print(snippet)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    if changes_made:
        print(f"✅ DONE — {len(changes_made)} change(s) applied:")
        for i, change in enumerate(changes_made, 1):
            print(f"   {i}. {change}")
    else:
        print("✅ DONE — No changes were needed. Everything is already set up.")
    print("=" * 60)


if __name__ == "__main__":
    setup_dev_env()
