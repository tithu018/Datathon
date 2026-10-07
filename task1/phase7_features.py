"""Task 1 Phase 7: add planned-information feature groups cumulatively to the Phase 6
features, with the SAME Random Forests, and keep a group only if it helps.

Primary metric: the time-based test (order_date >= 2026-01-03); random split alongside.
Keep rule (time-based): the group lowers service MAE or late log loss by at least
IMPROVE (relative), and neither of the two gets worse by more than HURT (relative).
Rejected groups are not carried into later steps.

Also reports: late rate by month and monsoon flag, road_conditions.csv coverage and
contents (not used as a feature), and one labelled extra (cum_unplanned_wait_before)
that is measured but never kept automatically.

Run from the repo root after task1/prepare_features.py. Writes
task1/reports/phase7_feature_groups.csv and task1/reports/metrics_phase7.json.
"""
import json
import time
from pathlib import Path

import pandas as pd

from evaluation import TIME_CUTOFF, metrics, random_split, time_split
from features import BASE_FEATURES, EXTRA_CANDIDATES, FEATURE_GROUPS, assert_no_leakage
from models import make_rf_models

IMPROVE, HURT = 0.005, 0.01
REPORTS = Path("task1") / "reports"
df = pd.read_csv(Path("task1") / "train_features.csv")
labels = pd.read_csv(Path("task1") / "task1_training_labels.csv", usecols=["delivery_id", "date"])
assert (labels["delivery_id"] == df["delivery_id"]).all()
SPLITS = {"time": time_split(df), "random": random_split(df)}


def evaluate(features):
    assert_no_leakage(features)
    missing = [c for c in features if c not in df.columns]
    assert not missing, missing
    out = {}
    for name, (tr, te) in SPLITS.items():
        s, l = make_rf_models(features)
        s.fit(df.loc[tr, features], df.loc[tr, "service_minutes"])
        l.fit(df.loc[tr, features], df.loc[tr, "late"])
        out[name] = metrics(df.loc[te, "service_minutes"], s.predict(df.loc[te, features]),
                            df.loc[te, "late"], l.predict_proba(df.loc[te, features])[:, 1])
    return out


def row(step, group, added, res, kept, ref=None):
    t, r = res["time"], res["random"]
    d = {"step": step, "group": group, "added": ", ".join(added), "kept": kept,
         "time_mae": t["service_mae"], "time_rmse": t["service_rmse"], "time_auc": t["late_roc_auc"],
         "time_logloss": t["late_log_loss"], "time_brier": t["late_brier"], "time_share_p0": t["late_share_exact_0"],
         "random_mae": r["service_mae"], "random_auc": r["late_roc_auc"], "random_logloss": r["late_log_loss"]}
    if ref is not None:
        d["d_time_mae_pct"] = round(100 * (t["service_mae"] / ref["time"]["service_mae"] - 1), 2)
        d["d_time_logloss_pct"] = round(100 * (t["late_log_loss"] / ref["time"]["late_log_loss"] - 1), 2)
    return d


t0 = time.time()
kept, rows = list(BASE_FEATURES), []
best = evaluate(kept)
rows.append(row(0, "Phase 6 base (17 features)", [], best, "base"))
print(f"base: time MAE {best['time']['service_mae']}, log loss {best['time']['late_log_loss']} ({time.time() - t0:.0f}s)")
kept_groups = []
for i, (group, cols) in enumerate(FEATURE_GROUPS.items(), start=1):
    res = evaluate(kept + cols)
    d_mae = res["time"]["service_mae"] / best["time"]["service_mae"] - 1
    d_ll = res["time"]["late_log_loss"] / best["time"]["late_log_loss"] - 1
    keep = (d_mae <= -IMPROVE or d_ll <= -IMPROVE) and d_mae <= HURT and d_ll <= HURT
    rows.append(row(i, group, cols, res, "yes" if keep else "no", best))
    print(f"{group}: time MAE {d_mae:+.2%}, log loss {d_ll:+.2%} -> {'KEEP' if keep else 'reject'} "
          f"({time.time() - t0:.0f}s)")
    if keep:
        kept, best = kept + cols, res
        kept_groups.append(group)

# Labelled extra, measured on top of the kept set; never kept automatically
for name, cols in EXTRA_CANDIDATES.items():
    res = evaluate(kept + cols)
    rows.append(row("extra", f"{name} (EXTRA, user decision)", cols, res, "not applied", best))
    print(f"extra {name}: time MAE {res['time']['service_mae'] / best['time']['service_mae'] - 1:+.2%}, "
          f"log loss {res['time']['late_log_loss'] / best['time']['late_log_loss'] - 1:+.2%}")
table = pd.DataFrame(rows)
table.to_csv(REPORTS / "phase7_feature_groups.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 40)
print("\n=== Feature groups (cumulative; time-based test primary) ===")
print(table.drop(columns="added").to_string(index=False))

# ---------------------------------------------------------------- late rate by month / monsoon
od = pd.to_datetime(df["order_date"])
late = df.assign(month=od.dt.month, year=od.dt.year, period=(od >= pd.Timestamp(TIME_CUTOFF)).map(
    {True: "time test", False: "time train"}))
by_month = late.pivot_table(index="month", columns="year", values="late", aggfunc="mean").round(4)
by_monsoon = late.groupby("monsoon")["late"].agg(["mean", "size"]).round(4)
by_period = late.groupby("period")["late"].agg(["mean", "size"]).round(4)
same_months = late[late["month"].isin([1, 2])].groupby("year")["late"].mean().round(4)
print("\n=== Late rate by month (rows) and year ===")
print(by_month.to_string())
print("\nBy monsoon flag:", by_monsoon.to_dict("index"))
print("Time-based train vs test:", by_period.to_dict("index"))
print("January-February by year:", same_months.to_dict())
by_brand_period = late.pivot_table(index="brand", columns="period", values="late", aggfunc="mean").round(4)
print("By brand and period:\n" + by_brand_period.to_string())

# ---------------------------------------------------------------- road conditions (report only)
rc = pd.read_csv(Path("General Data") / "road_conditions.csv")
test_legs = pd.read_csv(Path("Test Data") / "route_legs_test.csv")
test_keys = test_legs[["district", "date"]].drop_duplicates()
cov = test_keys.merge(rc, on=["district", "date"], how="left")
train_rc = df[["district", "late", "service_minutes"]].assign(date=labels["date"]).merge(rc, on=["district", "date"], how="left")
train_rc["bucket"] = pd.cut(train_rc["disruption_index"], [0, 60, 80, 99, 100], labels=["<=60", "61-80", "81-99", "100"])
rc_report = {
    "columns": rc.columns.tolist(),
    "rows": len(rc), "date_range": [rc["date"].min(), rc["date"].max()], "districts": int(rc["district"].nunique()),
    "disruption_index": rc["disruption_index"].describe().round(2).to_dict(),
    "share_days_below_100": round(float((rc["disruption_index"] < 100).mean()), 4),
    "test_dates": [test_legs["date"].min(), test_legs["date"].max()],
    "test_district_dates": int(len(test_keys)), "test_district_dates_covered": int(cov["disruption_index"].notna().sum()),
    "test_share_below_100": round(float((cov["disruption_index"] < 100).mean()), 4),
    "train_late_rate_by_bucket": train_rc.groupby("bucket", observed=True)["late"].mean().round(4).to_dict(),
    "train_service_mean_by_bucket": train_rc.groupby("bucket", observed=True)["service_minutes"].mean().round(2).to_dict(),
    "train_rows_by_bucket": train_rc.groupby("bucket", observed=True).size().to_dict(),
}
print("\n=== road_conditions.csv (reported only, NOT used) ===")
for k, v in rc_report.items():
    print(f"  {k}: {v}")

final = best
(REPORTS / "metrics_phase7.json").write_text(json.dumps({
    "description": "Phase 7: Phase 6 features + kept planned-information groups; same Random Forests",
    "keep_rule": {"improve_rel": IMPROVE, "hurt_rel": HURT, "primary": "time-based test"},
    "kept_groups": kept_groups,
    "features": kept,
    "random_split": final["random"],
    "time_based": {"cutoff_order_date": TIME_CUTOFF, **final["time"]},
    "groups": rows,
    "late_rate": {"by_month_year": {str(c): by_month[c].dropna().to_dict() for c in by_month.columns},
                  "by_monsoon": by_monsoon.to_dict("index"), "by_period": by_period.to_dict("index"),
                  "jan_feb_by_year": same_months.to_dict(), "by_brand_period": by_brand_period.to_dict()},
    "road_conditions": rc_report,
}, indent=2, default=str))
print(f"\nKept groups: {kept_groups}\nSaved: {REPORTS / 'phase7_feature_groups.csv'}, {REPORTS / 'metrics_phase7.json'} "
      f"({time.time() - t0:.0f}s)")
