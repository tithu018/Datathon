"""In-sample check of the final Fresh model where a payday window overlaps a festival ramp
(the 2026-W22 situation: paydays May 25 and May 30 plus the Poson ramp).

Uses the saved final models (task2a/models/task2a_model.joblib) on the training history
only. Ratio = actual / fitted Fresh daily volume (0.6 GLM + 0.4 HGB, as in the final config).
Run from the repo root after task2a/train_model.py.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA = Path("task2a") / "data"
REPORTS = Path("task2a") / "reports"
bundle = joblib.load(Path("task2a") / "models" / "task2a_model.joblib")
fresh = bundle["fitted"]["brands"]["Fresh"]
assert fresh["spec"]["model"] == "blend"

daily = pd.read_csv(DATA / "daily_features.csv", parse_dates=["date"])
h = daily[(daily["split"] == "history") & (daily["brand"] == "Fresh")].copy()
fitted = np.zeros(len(h))
for part, w in fresh["parts"]:
    if part["spec"]["model"] == "glm":
        p = np.zeros(len(h))
        for (d, b), m in part["models"].items():
            mask = (h["depot"] == d).to_numpy()
            p[mask] = m.predict(h[mask])
    else:
        p = part["model"].predict(h)
    fitted += w * p
h["fitted"] = fitted

op = h[h["is_operating"] == 1].copy()
op["payday_window"] = op[["payday_cal_d0", "payday_cal_d1", "payday_cal_d2"]].max(axis=1) == 1
op["ramp"] = op["festival_ramp"] > 0
op["group"] = np.select([op["payday_window"] & op["ramp"], op["payday_window"], op["ramp"]],
                        ["payday + ramp (overlap)", "payday only", "ramp only"], "neither")
op["ratio"] = op["total_volume"] / op["fitted"]
g = op.groupby("group").agg(series_days=("ratio", "size"), mean_daily_ratio=("ratio", "mean"),
                            sum_actual=("total_volume", "sum"), sum_fitted=("fitted", "sum"))
g["sum_ratio"] = g["sum_actual"] / g["sum_fitted"]
g = g.loc[["payday + ramp (overlap)", "payday only", "ramp only", "neither"]]
print("Fresh, operating days, in-sample (both depots): actual / fitted")
print(g.round(4).to_string())
ov = op[op["group"] == "payday + ramp (overlap)"]
print("\nOverlap days by festival:")
print(ov.groupby("festival_name").agg(series_days=("ratio", "size"), mean_daily_ratio=("ratio", "mean"),
                                       sum_ratio=("total_volume", lambda s: s.sum() / ov.loc[s.index, "fitted"].sum()))
      .round(4).to_string())

# Weekly view: Fresh both depots, actual vs fitted
cal = pd.read_csv(DATA / "calendar_features.csv")
wk_cal = cal.groupby(["iso_year", "iso_week"]).agg(paydays=("is_payday", "sum"), max_ramp=("festival_ramp", "max"),
                                                   festivals=("festival_name", lambda s: ",".join(sorted(set(s) - {"none"}))),
                                                   op_days=("is_operating", "sum"))
wk = h.groupby(["iso_year", "iso_week"]).agg(actual=("total_volume", "sum"), fitted=("fitted", "sum")).join(wk_cal)
wk["ratio"] = wk["actual"] / wk["fitted"]
two = wk[wk["paydays"] >= 2]
both = wk[(wk["paydays"] >= 1) & (wk["max_ramp"] > 0)]
pd.set_option("display.width", 200)
print(f"\nHistory weeks with 2 paydays ({len(two)}), Fresh both depots:")
print(two.round(3).to_string())
print(f"  mean weekly ratio {two['ratio'].mean():.4f}, sum ratio {two['actual'].sum() / two['fitted'].sum():.4f}")
print(f"\nHistory weeks with a payday and festival ramp > 0 ({len(both)}), Fresh both depots:")
print(both.round(3).to_string())
print(f"  mean weekly ratio {both['ratio'].mean():.4f}, sum ratio {both['actual'].sum() / both['fitted'].sum():.4f}")
ratio = g.loc["payday + ramp (overlap)", "mean_daily_ratio"]
print(f"\nDecision rule: overlap-day mean ratio {ratio:.4f} -> "
      f"{'within [0.95, 1.05]: keep the forecast' if 0.95 <= ratio <= 1.05 else 'OUTSIDE [0.95, 1.05]: stop and report'}")
