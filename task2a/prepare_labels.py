"""Task 2A labels: daily and weekly order volume per depot x brand.

Rules (Challenge Booklet, Task 2A):
- every order counts once, including deferred and not_run orders (they are demand);
- each order is assigned to the week the store requested it (order_date, never
  dispatch_date), using iso_year / iso_week from calendar.csv;
- only Fresh has chilled demand.

History = Training Data/deliveries_train.csv + Test Data/task1_test_inputs.csv.
Run from the repo root. Writes task2a/data/daily_volume.csv and
task2a/data/weekly_volume.csv (gitignored).
"""
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_ORDERS = Path("Training Data") / "deliveries_train.csv"
TASK1_ORDERS = Path("Test Data") / "task1_test_inputs.csv"
CALENDAR = Path("General Data") / "calendar.csv"
OUTLETS = Path("General Data") / "outlets.csv"
OUT_DIR = Path("task2a") / "data"

DEPOTS = ["Kandy", "Peliyagoda"]
BRANDS = ["Fresh", "Style", "Tech"]

# Independently computed reference numbers (user, Phase 1 brief)
REF = {
    "n_train": 92_307,
    "n_task1": 5_014,
    "n_total": 97_321,
    "first_order_date": "2024-01-01",
    "last_order_date": "2026-03-28",
    "status": {"attempted": 95_275, "deferred": 1_633, "not_run": 413},
    "volume": {
        ("Fresh", "ambient"): 107_905.754,
        ("Fresh", "chilled"): 62_453.090,
        ("Fresh", "total"): 170_358.844,
        ("Style", "total"): 26_044.960,
        ("Tech", "total"): 5_963.394,
    },
    "n_weeks": 117,
    "first_week": (2024, 1),
    "last_week": (2026, 13),
    "mean_weekly": {
        ("Kandy", "Fresh"): 500.26, ("Kandy", "Style"): 80.94, ("Kandy", "Tech"): 21.19,
        ("Peliyagoda", "Fresh"): 955.80, ("Peliyagoda", "Style"): 141.67, ("Peliyagoda", "Tech"): 29.78,
    },
    "pel_fresh_2025_w15": 1_351.0,
    "pel_fresh_2025_w16": 623.5,
}


def check(name, actual, expected, tol=0.0):
    """Print and assert one reference check."""
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        ok = abs(actual - expected) <= tol
    else:
        ok = actual == expected
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {actual} (reference {expected})")
    assert ok, f"{name}: got {actual}, expected {expected}"


# ---------------------------------------------------------------- load
train = pd.read_csv(TRAIN_ORDERS)
task1 = pd.read_csv(TASK1_ORDERS)
calendar = pd.read_csv(CALENDAR)
outlets = pd.read_csv(OUTLETS)

print("1. Orders")
check("train orders", len(train), REF["n_train"])
check("task1 orders", len(task1), REF["n_task1"])
overlap = set(train["delivery_id"]) & set(task1["delivery_id"])
check("overlapping delivery_id", len(overlap), 0)

train["source"] = "train"
task1["source"] = "task1"
cols = ["delivery_id", "order_date", "dispatch_status", "outlet_id", "brand", "depot",
        "temp_requirement", "order_volume_m3", "source"]
orders = pd.concat([train[cols], task1[cols]], ignore_index=True)
check("combined orders", len(orders), REF["n_total"])
assert orders["delivery_id"].is_unique
assert orders["order_volume_m3"].notna().all() and (orders["order_volume_m3"] > 0).all()

# Outlet attributes in the order files must agree with outlets.csv
o = orders.merge(outlets[["outlet_id", "brand", "depot"]], on="outlet_id", how="left", suffixes=("", "_outlet"))
assert o["brand_outlet"].notna().all(), "order with unknown outlet_id"
assert (o["brand"] == o["brand_outlet"]).all() and (o["depot"] == o["depot_outlet"]).all()
assert set(orders["depot"]) == set(DEPOTS) and set(orders["brand"]) == set(BRANDS)
print("  [PASS] brand/depot of every order match outlets.csv")

# ---------------------------------------------------------------- dates and weeks
print("\n2. Order dates and calendar")
check("first order_date", orders["order_date"].min(), REF["first_order_date"])
check("last order_date", orders["order_date"].max(), REF["last_order_date"])
cal_cols = ["date", "iso_year", "iso_week", "is_operating"]
orders = orders.merge(calendar[cal_cols], left_on="order_date", right_on="date", how="left")
check("orders without a calendar date", int(orders["iso_year"].isna().sum()), 0)
orders = orders.drop(columns="date")
check("orders on is_operating = 0 days", int((orders["is_operating"] == 0).sum()), 0)

# ---------------------------------------------------------------- status
print("\n3. dispatch_status (combined; every status is kept as demand)")
status = orders["dispatch_status"].value_counts()
for s, n in REF["status"].items():
    check(f"status {s}", int(status.get(s, 0)), n)
assert status.sum() == len(orders), "unexpected dispatch_status value"
print("  By source:")
print(pd.crosstab(orders["source"], orders["dispatch_status"], margins=True).to_string().replace("\n", "\n    "))

# Caveat: the Task 1 file only holds dispatched orders (no not_run)
tr = orders[orders["source"] == "train"]
not_run_share = tr.loc[tr["dispatch_status"] == "not_run", "order_volume_m3"].sum() / tr["order_volume_m3"].sum()
print(f"  Caveat: not_run orders are {not_run_share:.2%} of training-file volume. The task1 file has "
      f"{int((orders['source'].eq('task1') & orders['dispatch_status'].eq('not_run')).sum())} not_run orders, "
      f"so its 6 weeks (2026-W08..W13) may undercount demand by about that share.")

# ---------------------------------------------------------------- volume by brand
print("\n4. Total order_volume_m3 by brand")
vol = orders.groupby(["brand", "temp_requirement"])["order_volume_m3"].sum()
check("Fresh ambient", round(vol[("Fresh", "ambient")], 3), REF["volume"][("Fresh", "ambient")], 5e-4)
check("Fresh chilled", round(vol[("Fresh", "chilled")], 3), REF["volume"][("Fresh", "chilled")], 5e-4)
for b in BRANDS:
    check(f"{b} total", round(vol[b].sum(), 3), REF["volume"][(b, "total")], 5e-4)
chilled_brands = sorted(orders.loc[orders["temp_requirement"] == "chilled", "brand"].unique())
check("brands with chilled orders", chilled_brands, ["Fresh"])

# ---------------------------------------------------------------- daily grid
print("\n5. Daily grid (calendar date x depot x brand, zero-filled)")
orders["chilled_volume"] = np.where(orders["temp_requirement"] == "chilled", orders["order_volume_m3"], 0.0)
daily_obs = (orders.groupby(["order_date", "depot", "brand"])
             .agg(total_volume=("order_volume_m3", "sum"),
                  chilled_volume=("chilled_volume", "sum"),
                  n_orders=("delivery_id", "size"))
             .reset_index().rename(columns={"order_date": "date"}))

# Cover complete ISO weeks: from the Monday of the first order week to the Sunday of the last
weeks = calendar[["iso_year", "iso_week"]].apply(tuple, axis=1)
first_wk = tuple(orders.loc[orders["order_date"].idxmin(), ["iso_year", "iso_week"]].astype(int))
last_wk = tuple(orders.loc[orders["order_date"].idxmax(), ["iso_year", "iso_week"]].astype(int))
cal_hist = calendar[(weeks >= first_wk) & (weeks <= last_wk)][cal_cols]
grid = cal_hist.merge(pd.DataFrame([(d, b) for d in DEPOTS for b in BRANDS], columns=["depot", "brand"]), how="cross")
daily = grid.merge(daily_obs, on=["date", "depot", "brand"], how="left")
daily[["total_volume", "chilled_volume"]] = daily[["total_volume", "chilled_volume"]].fillna(0.0)
daily["n_orders"] = daily["n_orders"].fillna(0).astype(int)
daily = daily.sort_values(["depot", "brand", "date"]).reset_index(drop=True)

assert len(daily_obs) == len(daily_obs.merge(cal_hist, on="date")), "orders outside the grid"
days_per_week = daily.groupby(["depot", "brand", "iso_year", "iso_week"]).size()
assert (days_per_week == 7).all(), "incomplete week in the daily grid"
print(f"  {len(daily):,} rows = {cal_hist['date'].nunique()} days x 6 series; "
      f"dates {cal_hist['date'].min()} to {cal_hist['date'].max()}")

# ---------------------------------------------------------------- weekly table
print("\n6. Weekly table (depot x brand x iso_year x iso_week)")
weekly = (daily.groupby(["depot", "brand", "iso_year", "iso_week"])
          .agg(total_volume=("total_volume", "sum"),
               chilled_volume=("chilled_volume", "sum"),
               n_orders=("n_orders", "sum"),
               operating_days=("is_operating", "sum"),
               week_start=("date", "min"))
          .reset_index())

n_weeks = weekly.groupby(["depot", "brand"]).size()
check("number of series", len(n_weeks), 6)
for (d, b), n in n_weeks.items():
    check(f"weeks {d} {b}", int(n), REF["n_weeks"])
week_keys = sorted(weekly[["iso_year", "iso_week"]].drop_duplicates().itertuples(index=False, name=None))
check("first week", tuple(map(int, week_keys[0])), REF["first_week"])
check("last week", tuple(map(int, week_keys[-1])), REF["last_week"])

raw_total = orders["order_volume_m3"].sum()
raw_chilled = orders["chilled_volume"].sum()
check("daily total == raw total", round(daily["total_volume"].sum(), 6), round(raw_total, 6), 1e-6)
check("weekly total == raw total", round(weekly["total_volume"].sum(), 6), round(raw_total, 6), 1e-6)
check("daily chilled == raw chilled", round(daily["chilled_volume"].sum(), 6), round(raw_chilled, 6), 1e-6)
check("weekly chilled == raw chilled", round(weekly["chilled_volume"].sum(), 6), round(raw_chilled, 6), 1e-6)
check("weekly n_orders == combined orders", int(weekly["n_orders"].sum()), REF["n_total"])
check("chilled volume outside Fresh", float(weekly.loc[weekly["brand"] != "Fresh", "chilled_volume"].abs().sum()), 0.0)
check("volume on non-operating days", float(daily.loc[daily["is_operating"] == 0, "total_volume"].sum()), 0.0)

print("\n7. Mean weekly total volume per series")
mean_weekly = weekly.groupby(["depot", "brand"])["total_volume"].mean()
for key, ref in REF["mean_weekly"].items():
    check(f"mean weekly {key[0]} {key[1]}", round(mean_weekly[key], 2), ref, 0.005)

print("\n8. Sanity check: Peliyagoda Fresh around New Year 2025")
pf = weekly[(weekly["depot"] == "Peliyagoda") & (weekly["brand"] == "Fresh")].set_index(["iso_year", "iso_week"])
check("2025-W15 (build-up)", round(pf.loc[(2025, 15), "total_volume"], 1), REF["pel_fresh_2025_w15"], 0.5)
check("2025-W16 (festival week)", round(pf.loc[(2025, 16), "total_volume"], 1), REF["pel_fresh_2025_w16"], 0.5)
print(f"  operating days: W15 = {pf.loc[(2025, 15), 'operating_days']}, W16 = {pf.loc[(2025, 16), 'operating_days']}")

# ---------------------------------------------------------------- save
OUT_DIR.mkdir(parents=True, exist_ok=True)
daily.to_csv(OUT_DIR / "daily_volume.csv", index=False)
weekly.to_csv(OUT_DIR / "weekly_volume.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'daily_volume.csv'} ({len(daily):,} rows), "
      f"{OUT_DIR / 'weekly_volume.csv'} ({len(weekly):,} rows)")
print("All Phase 1 checks passed.")
