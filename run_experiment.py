import sys
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

import papermill as pm

# Create logs folder if it doesn't exist
log_dir = Path("annesnotes/logs")
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
executed_nb = log_dir / "Machine Learning_executed.ipynb"

with open(log_file, "w") as f:
    f.write(f"Starting experiment at {datetime.now()}\n")

try:
    with open(log_file, "a") as stdout_file:
        pm.execute_notebook(
            "Machine Learning.ipynb",
            str(executed_nb),
            log_output=True,
            stdout_file=stdout_file,
            start_timeout=120,
        )
    print(f"✓ Experiment completed successfully. Log: {log_file}")
    sys.exit(0)
except Exception as e:
    with open(log_file, "a") as f:
        f.write(f"\nExperiment failed: {e}\n")
    print(f"✗ Experiment failed. Check log: {log_file}")
    sys.exit(1)

#to run: python run_experiment.py
#to monitor log while its running, in another terminal: Get-Content annesnotes/logs/experiment_log_*.txt -Wait