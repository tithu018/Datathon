"""Compare the regenerated Task 1 submission with the original (eaf7297) submission:
distributions, correlations, mean predicted late rate, and the 10 largest changes with
the planned features that explain them. Writes task1/reports/submission_comparison.txt.
Run from the repo root after task1/make_submission.py."""
from pathlib import Path

import pandas as pd

REPORTS = Path("task1") / "reports"
orig = pd.read_csv(REPORTS / "submission_task1_original.csv")
new = pd.read_csv(Path("outputs") / "submission_task1.csv")
assert orig["delivery_id"].tolist() == new["delivery_id"].tolist()
test = pd.read_csv(Path("task1") / "test_features.csv")
inputs = pd.read_csv(Path("Test Data") / "task1_test_inputs.csv", usecols=["delivery_id", "order_date"])
ctx = test.merge(inputs, on="delivery_id")[["delivery_id", "order_date", "brand", "district", "monsoon",
                                            "disruption_index", "festival_ramp", "is_payday", "order_units",
                                            "dock_type", "slack_to_close_min", "seq"]]
d = orig.merge(new, on="delivery_id", suffixes=("_orig", "_new")).merge(ctx, on="delivery_id")
lines = []


def out(s=""):
    print(s)
    lines.append(str(s))


pd.set_option("display.width", 250)
q = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
for col in ("pred_service_min", "pred_late_prob"):
    t = pd.DataFrame({"original": d[f"{col}_orig"].describe(percentiles=q), "new": d[f"{col}_new"].describe(percentiles=q)})
    out(f"=== {col} distribution ===")
    out(t.round(4).to_string())
    out(f"Pearson {d[f'{col}_orig'].corr(d[f'{col}_new']):.4f}, Spearman "
        f"{d[f'{col}_orig'].corr(d[f'{col}_new'], method='spearman'):.4f}")
    out()
out(f"Share of exact 0 late probability: original {(d['pred_late_prob_orig'] == 0).mean():.2%}, new "
    f"{(d['pred_late_prob_new'] == 0).mean():.2%}")
out("Mean predicted late rate: original {:.4f}, new {:.4f}".format(d["pred_late_prob_orig"].mean(),
                                                                     d["pred_late_prob_new"].mean()))
out("By month and monsoon flag (new | original):")
d["month"] = d["order_date"].str[:7]
out(d.groupby(["month", "monsoon"])[["pred_late_prob_new", "pred_late_prob_orig"]].mean().round(4).to_string())
out("By brand (new | original):")
out(d.groupby("brand")[["pred_late_prob_new", "pred_late_prob_orig", "pred_service_min_new",
                        "pred_service_min_orig"]].mean().round(3).to_string())
out("By disruption_index bucket (new | original):")
d["disruption"] = pd.cut(d["disruption_index"], [0, 60, 80, 99, 100], labels=["<=60", "61-80", "81-99", "100"])
out(d.groupby("disruption", observed=True)[["pred_late_prob_new", "pred_late_prob_orig"]].agg(["mean", "size"]).round(4).to_string())
out()

cols = ["delivery_id", "order_date", "brand", "district", "monsoon", "disruption_index", "festival_ramp", "is_payday",
        "order_units", "dock_type", "slack_to_close_min"]
for col, unit in (("pred_late_prob", ""), ("pred_service_min", " min")):
    d["change"] = d[f"{col}_new"] - d[f"{col}_orig"]
    top = d.reindex(d["change"].abs().sort_values(ascending=False).index).head(10)
    out(f"=== 10 largest changes in {col} ===")
    out(top[cols + [f"{col}_orig", f"{col}_new", "change"]].round(3).to_string(index=False))
    out()
(REPORTS / "submission_comparison.txt").write_text("\n".join(lines))
print("Saved: task1/reports/submission_comparison.txt")
