from pathlib import Path

def find_thesis_root(marker="settings.json"):
    """Walk up from cwd until we find the marker file."""
    current = Path().resolve()
    for parent in [current, *current.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find thesis root (looked for '{marker}')")