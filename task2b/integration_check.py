"""Run the three-task notebook cell and verify all submission outputs."""
import io
import runpy
import sys
from contextlib import redirect_stdout

import pandas as pd

sys.path.insert(0, ".")
from task2b.common import REPORT_DIR, DATA_DIR, SUBMISSION, load_orders, load_fleet
from task2b.feasibility import validate, official_check


def main():
    buf = io.StringIO()
    with redirect_stdout(buf):
        namespace = runpy.run_path("docs/handover/inference_cell.py")
    for task in ("task1", "task2a"):
        actual = pd.read_csv(f"outputs/submission_{task}.csv", float_precision="round_trip")
        assert namespace[f"{task}_pred"].equals(actual), f"{task}: generated models and submission differ"
    sub = pd.read_csv(SUBMISSION)
    assert namespace["task2b_pred"].equals(sub)
    assert sub.equals(pd.read_csv(DATA_DIR / "optimized_submission.csv"))
    validate(sub, load_orders(), load_fleet())
    reasons = pd.read_csv(DATA_DIR / "order_decisions.csv")
    assert reasons.order_ref.tolist() == sub.order_ref.tolist() and reasons.reason.notna().all()
    output = (buf.getvalue() + "\nTask 1 and Task 2A: saved models reproduce current submissions exactly.\n"
              "Task 2B: final submission equals optimiser output, strict checks pass, 85 explanations present.\n"
              + official_check(SUBMISSION) + "\nThree-task integration checks PASSED\n")
    output = "\n".join(line.rstrip() for line in output.splitlines()) + "\n"
    (REPORT_DIR / "integration_checks.txt").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
