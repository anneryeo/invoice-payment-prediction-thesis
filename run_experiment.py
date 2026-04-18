import os
import sys
from datetime import datetime
from pathlib import Path

# Set OpenBLAS thread limit to suppress warning
os.environ["OPENBLAS_NUM_THREADS"] = "24"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

import papermill as pm


class _Tee:
    """Writes to both the terminal and the log file simultaneously."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


# Create logs folder if it doesn't exist
log_dir = Path("annesnotes/logs")
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
executed_nb = log_dir / "Machine Learning_executed.ipynb"

print(f"Starting experiment at {datetime.now()}")
print(f"Log: {log_file}\n")

try:
    with open(log_file, "w", encoding="utf-8") as log_f:
        tee = _Tee(sys.stdout, log_f)
        pm.execute_notebook(
            "Machine Learning.ipynb",
            str(executed_nb),
            log_output=True,
            stdout_file=tee,
            start_timeout=120,
        )
    print(f"\n✓ Experiment completed successfully.")
    sys.exit(0)
except Exception as e:
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\nExperiment failed: {e}\n")
    print(f"\n✗ Experiment failed: {e}")
    print(f"Check log: {log_file}")
    sys.exit(1)

#to run: python run_experiment.py
#to monitor log while its running, in another terminal: Get-Content annesnotes/logs/experiment_log_*.txt -Wait