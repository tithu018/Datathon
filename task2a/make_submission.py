"""Task 2A submission: fill Submission Templates/submission_task2a.csv from predict_task2a
and write outputs/submission_task2a.csv. Run from the repo root after task2a/train_model.py."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import OUTPUT_COLUMNS, predict_task2a  # noqa: E402

TEMPLATE = Path("Submission Templates") / "submission_task2a.csv"
INPUTS = Path("Test Data") / "task2a_test_inputs.csv"
OUT = Path("outputs") / "submission_task2a.csv"
DECIMALS = 3

template = pd.read_csv(TEMPLATE)
inputs = pd.read_csv(INPUTS)
assert list(template.columns) == OUTPUT_COLUMNS, template.columns
assert template["row_id"].is_unique and set(template["row_id"]) == set(inputs["row_id"])

pred = predict_task2a(inputs).set_index("row_id")
sub = template[["row_id"]].copy()
for c in OUTPUT_COLUMNS[1:]:
    sub[c] = sub["row_id"].map(pred[c]).round(DECIMALS)

# ---------------------------------------------------------------- checks
brand = sub["row_id"].map(inputs.set_index("row_id")["brand"])
assert len(sub) == 60, len(sub)
assert sub["row_id"].tolist() == template["row_id"].tolist(), "row_id values/order differ from the template"
assert sub[OUTPUT_COLUMNS[1:]].notna().all().all(), "missing prediction"
assert (sub[OUTPUT_COLUMNS[1:]] >= 0).all().all(), "negative prediction"
assert (sub.loc[brand != "Fresh", "pred_chilled_volume_m3"] == 0).all(), "chilled must be 0 for Style/Tech"
assert (sub["pred_chilled_volume_m3"] <= sub["pred_total_volume_m3"]).all(), "chilled exceeds total"
assert (sub.loc[brand == "Fresh", "pred_chilled_volume_m3"] > 0).all()

OUT.parent.mkdir(parents=True, exist_ok=True)
sub.to_csv(OUT, index=False)
print(f"Task 2A submission: {len(sub)} rows, all checks passed. Saved: {OUT}")
print(sub.assign(brand=brand.values).groupby("brand")[OUTPUT_COLUMNS[1:]].sum().round(1).to_string())
