"""Run the whole Task 1 pipeline in order, from raw data to outputs/submission_task1.csv.

    python task1/run_all.py            # labels -> features -> validation + final fit -> submission -> comparison
    python task1/run_all.py --no-eval  # skip the validation runs in train_model.py (final fit only)

Run from the repo root. Stops at the first failing step.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
train_args = ["--no-eval"] if "--no-eval" in sys.argv else []
STEPS = [("prepare_labels.py",), ("prepare_features.py",), ("train_model.py", *train_args),
         ("make_submission.py",), ("compare_submission.py",)]

for step in STEPS:
    t0 = time.time()
    print(f"=== {' '.join(step)}", flush=True)
    r = subprocess.run([sys.executable, str(HERE / step[0]), *step[1:]], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
        sys.exit(f"step failed: {' '.join(step)}")
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    print(f"    ok ({time.time() - t0:.0f}s): {last}", flush=True)
print("Task 1 pipeline complete: outputs/submission_task1.csv")
