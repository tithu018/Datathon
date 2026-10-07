"""Phase 7: exact template output, independently checked after disk round-trip."""
import sys

import pandas as pd

sys.path.insert(0, ".")
from task2b.common import DATA_DIR, REPORT_DIR, SUBMISSION, load_orders, load_fleet
from task2b.feasibility import validate, official_check


def main():
    orders, fleet = load_orders(), load_fleet()
    candidate = pd.read_csv(DATA_DIR / "optimized_submission.csv")
    validate(candidate, orders, fleet)
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    # Preserve literal integer trip IDs and blank fields, not float CSV tokens.
    SUBMISSION.write_bytes((DATA_DIR / "optimized_submission.csv").read_bytes())
    saved = pd.read_csv(SUBMISSION)
    assert saved.equals(candidate)
    minutes = validate(saved, orders, fleet)
    output = (f"Final Task 2B submission: {SUBMISSION}\n"
              f"85 template rows and identifiers preserved; served {saved.decision.eq('served').sum()}, "
              f"deferred {saved.decision.eq('deferred').sum()}; {minutes} published minutes.\n"
              + official_check(SUBMISSION) + "\n")
    (REPORT_DIR / "final_checker_output.txt").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
