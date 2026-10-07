"""Run the whole Task 2B pipeline in order.

    python task2b/run_all.py

Run from the repo root. Stops at the first failing step. Later phases add their steps
to STEPS (analysis, greedy allocation, optimiser, checks, submission).
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["smoke_check.py", "analysis.py", "policy.py", "allocate_greedy.py",
         "verify_optimizer.py", "optimize.py", "verify_timeline.py", "audit.py",
         "sensitivity.py", "writeup.py"]

for step in STEPS:
    t0 = time.time()
    print(f"=== {step}", flush=True)
    r = subprocess.run([sys.executable, str(HERE / step)], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
        sys.exit(f"step failed: {step}")
    print(f"    ok ({time.time() - t0:.0f}s): {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}")
print("Task 2B pipeline complete")
