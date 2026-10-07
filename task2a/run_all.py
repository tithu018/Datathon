"""Run the whole Task 2A pipeline in order, from raw data to outputs/submission_task2a.csv.

    python task2a/run_all.py           # labels -> features -> final fit -> submission -> sanity report
    python task2a/run_all.py --full    # also re-run the analysis, the Phase 3 backtest and the
                                       # Phase 4 score of the saved config (no selection, no confirmation)

Run from the repo root. Stops at the first failing step.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["prepare_labels.py", "prepare_features.py", "train_model.py", "make_submission.py", "forecast_report.py"]
FULL = [("analysis.py",), ("backtest.py",), ("experiments.py", "--score-config")]

steps = [(s,) for s in STEPS]
if "--full" in sys.argv:
    steps = steps[:2] + FULL + steps[2:]
for step in steps:
    t0 = time.time()
    print(f"=== {' '.join(step)}", flush=True)
    r = subprocess.run([sys.executable, str(HERE / step[0]), *step[1:]], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
        sys.exit(f"step failed: {' '.join(step)}")
    print(f"    ok ({time.time() - t0:.0f}s): {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}")
print("Task 2A pipeline complete: outputs/submission_task2a.csv")
