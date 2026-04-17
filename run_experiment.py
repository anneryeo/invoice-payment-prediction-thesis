import subprocess
import sys
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

# Create logs folder if it doesn't exist
log_dir = Path("annesnotes/logs")
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
executed_nb = log_dir / "Machine Learning_executed.ipynb"

with open(log_file, "w") as f:
    f.write(f"Starting experiment at {datetime.now()}\n")

result = subprocess.run(
    [
        "papermill",
        "Machine Learning.ipynb",
        str(executed_nb),
        "--log-output",
        "--stdout-file", str(log_file)
    ],
    capture_output=False,
    text=True
)

if result.returncode == 0:
    print(f"✓ Experiment completed successfully. Log: {log_file}")
    sys.exit(0)
else:
    print(f"✗ Experiment failed. Check log: {log_file}")
    sys.exit(1)

#to run: python run_experiment.py
#to monitor log while its running, in another terminal: Get-Content annesnotes/logs/experiment_log_*.txt -Wait