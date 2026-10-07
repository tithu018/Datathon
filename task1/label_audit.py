"""Task 1 label audit (Phase 6). Read-only: checks the label construction in
task1/prepare_labels.py and reports; changes nothing.

Run from the repo root after task1/prepare_labels.py.
"""
from pathlib import Path

import pandas as pd

LABELS = Path("task1") / "task1_training_labels.csv"
TEST_ORDERS = Path("Test Data") / "task1_test_inputs.csv"
TEST_LEGS = Path("Test Data") / "route_legs_test.csv"

df = pd.read_csv(LABELS)
TIME_COLS = ["planned_arrival_time_order", "planned_arrival_time_leg", "planned_depart_time", "actual_depart_time",
             "arrival_time", "leave_outlet_time", "window_open_time", "window_close_time"]


def to_min(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, format="%H:%M", errors="coerce")
    return t.dt.hour * 60 + t.dt.minute


print("1. Time parsing")
print("   prepare_labels.py converts HH:MM with int(h)*60 + int(m) before every comparison and subtraction;")
print("   prepare_features.py uses pd.to_datetime(format='%H:%M', errors='coerce') -> minutes.")
m = {}
for c in TIME_COLS:
    m[c] = to_min(df[c])
    bad = int(m[c].isna().sum())
    fmt_ok = df[c].astype(str).str.fullmatch(r"\d{2}:\d{2}").all()
    print(f"   {c:28s} unparseable {bad}, all zero-padded HH:MM {fmt_ok}, range {df[c].min()}..{df[c].max()}")
# Re-derive the labels independently and compare
m = pd.DataFrame(m)
service = m["leave_outlet_time"] - m[["arrival_time", "window_open_time"]].max(axis=1)
late = (m["arrival_time"] > m["window_close_time"]).astype(int)
print(f"   independent re-derivation: service equal {bool((service == df['service_minutes']).all())}, "
      f"late equal {bool((late == df['late']).all())}")
# Would string comparison have given the same late flag? (only safe because all times are zero-padded)
late_str = (df["arrival_time"] > df["window_close_time"]).astype(int)
print(f"   string comparison would agree on late: {bool((late_str == df['late']).all())} (labels use minutes)")

print("\n2. Midnight crossing")
checks = {
    "actual arrival < actual departure": (m["arrival_time"] < m["actual_depart_time"]),
    "leave outlet < actual arrival": (m["leave_outlet_time"] < m["arrival_time"]),
    "planned arrival < planned departure": (m["planned_arrival_time_leg"] < m["planned_depart_time"]),
    "window close <= window open": (m["window_close_time"] <= m["window_open_time"]),
    "any time before 03:00": pd.concat([m[c] < 180 for c in TIME_COLS], axis=1).any(axis=1),
}
for k, v in checks.items():
    print(f"   {k:38s} {int(v.sum())}")
# Duration consistency: actual arrival = actual departure + actual travel
dur = m["actual_depart_time"] + df["actual_travel_duration_min"] - m["arrival_time"]
print(f"   arrival == actual_depart + actual_travel: {(dur == 0).mean():.2%} of legs (max |diff| {dur.abs().max()} min)")
pdur = m["planned_depart_time"] + df["planned_travel_duration_min"] - m["planned_arrival_time_leg"]
print(f"   planned arrival == planned_depart + planned_travel: {(pdur == 0).mean():.2%} of legs")
print(f"   order planned_arrival_time == leg planned_arrival_time: "
      f"{(df['planned_arrival_time_order'] == df['planned_arrival_time_leg']).mean():.2%}")

print("\n3. Early arrivals (vehicle waits until the window opens; service starts at window open)")
early = m["arrival_time"] < m["window_open_time"]
wait = (m["window_open_time"] - m["arrival_time"]).where(early)
print(f"   early arrivals: {int(early.sum()):,} of {len(df):,} ({early.mean():.2%}); wait median "
      f"{wait.median():.0f} min, max {wait.max():.0f} min")
print("   by brand:", df.assign(early=early).groupby("brand_order")["early"].agg(["sum", "mean"]).round(4).to_dict("index"))
naive = m["leave_outlet_time"] - m["arrival_time"]
print(f"   service if waiting were NOT removed: mean {naive.mean():.2f} min vs label mean {df['service_minutes'].mean():.2f} min")
print(f"   leave before window opens (impossible under the rule): {int((m['leave_outlet_time'] < m['window_open_time']).sum())}")

print("\n4. Lateness (arrival strictly after window close)")
print(f"   overall late rate {df['late'].mean():.4f}; arrivals exactly at window close (not late): "
      f"{int((m['arrival_time'] == m['window_close_time']).sum())}")
by = df.groupby("brand_order").agg(orders=("late", "size"), late_rate=("late", "mean"),
                                   service_mean=("service_minutes", "mean"), service_median=("service_minutes", "median"))
print(by.round(4).to_string())
print("   late rate by dispatch_status:", df.groupby("dispatch_status")["late"].mean().round(4).to_dict())

print("\n5. Other consistency")
same = {c: (df[f"{c}_order"] == df[f"{c}_leg"]).mean() for c in ["brand", "district", "depot", "vehicle_id",
                                                                   "vehicle_type", "vehicle_temp"]}
print("   order vs leg columns equal:", {k: f"{v:.2%}" for k, v in same.items()})
print(f"   leg date == dispatch_date: {(df['date'] == df['dispatch_date']).mean():.2%}")
print(f"   to_outlet == outlet_id: {(df['to_outlet'] == df['outlet_id']).mean():.2%}")
print(f"   service_minutes: min {df['service_minutes'].min()}, max {df['service_minutes'].max()}, "
      f"zeros {int((df['service_minutes'] == 0).sum())}")

print("\n6. Test inputs")
t = pd.read_csv(TEST_ORDERS).merge(pd.read_csv(TEST_LEGS), left_on=["route_id", "seq_in_route"],
                                   right_on=["route_id", "seq"], how="left", suffixes=("_order", "_leg"))
for c in ["planned_arrival_time_order", "planned_depart_time", "window_open_time", "window_close_time"]:
    print(f"   {c:28s} unparseable {int(to_min(t[c]).isna().sum())}")
print("   order vs leg columns equal:", {c: f"{(t[f'{c}_order'] == t[f'{c}_leg']).mean():.2%}"
                                         for c in ["brand", "district", "depot", "vehicle_type", "vehicle_temp"]})
print(f"   test planned arrival before window open: {(to_min(t['planned_arrival_time_order']) < to_min(t['window_open_time'])).mean():.2%}")
q = df["service_minutes"].quantile([0.5, 0.9, 0.99, 0.999])
print(f"\n7. Service-time tail: quantiles {q.round(0).to_dict()}; > 180 min: {int((df['service_minutes'] > 180).sum())} "
      f"({df[df['service_minutes'] > 180].groupby('brand_order').size().to_dict()}); 99th percentile by brand "
      f"{df.groupby('brand_order')['service_minutes'].quantile(0.99).round(0).to_dict()}. Kept (large Style/Tech orders).")
