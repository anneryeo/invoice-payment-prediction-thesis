import os
import sys
from datetime import datetime
from pathlib import Path

# Configure threading limits to prevent memory exhaustion and OpenBLAS warnings.
# OPENBLAS_NUM_THREADS=1 per worker: Pool provides process-level parallelism;
# >1 OpenBLAS threads per worker causes oversubscription and the
# "precompiled NUM_THREADS exceeded" warning on machines with >24 cores.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
# PYTHONWARNINGS is an OS-level env var read at interpreter startup by every
# spawned child process (including multiprocessing Pool workers on Windows),
# unlike warnings.filterwarnings() which only affects the current process.
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:sklearn.utils.parallel"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

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