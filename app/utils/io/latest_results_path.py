import os
import re

_DATE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}$")    # YYYY_MM_DD_##  where ## is the training run number for that day

def get_latest_results_path(RESULTS_ROOT) -> str:
    """Return the path to results.db inside the most-recent dated sub-folder."""
    dated_dirs = sorted(
        [d for d in os.listdir(RESULTS_ROOT) if _DATE_RE.match(d)],
        reverse=True,
    )
    if not dated_dirs:
        raise FileNotFoundError(f"No dated result folders found under {RESULTS_ROOT}")
    base = os.path.join(RESULTS_ROOT, dated_dirs[0])
    db_path = os.path.join(base, "results.db")
    if os.path.exists(db_path):
        return db_path
    raise FileNotFoundError(f"results.db not found in {base}")