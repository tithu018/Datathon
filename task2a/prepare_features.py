"""Task 2A daily features for history and the forecast window.

Calendar-level features are built for every date in calendar.csv, then crossed
with the 6 depot x brand series. History rows carry the Phase 1 labels
(total_volume, chilled_volume); forecast rows (2026-03-30 to 2026-06-07,
ISO weeks 14-23 of 2026) have empty labels.

Run from the repo root after task2a/prepare_labels.py. Writes
task2a/data/calendar_features.csv and task2a/data/daily_features.csv (gitignored).
"""
from pathlib import Path

import numpy as np
import pandas as pd

CALENDAR = Path("General Data") / "calendar.csv"
OUTLETS = Path("General Data") / "outlets.csv"
TASK2A_INPUTS = Path("Test Data") / "task2a_test_inputs.csv"
ORDER_FILES = [Path("Training Data") / "deliveries_train.csv", Path("Test Data") / "task1_test_inputs.csv"]
DATA = Path("task2a") / "data"

TREND_ORIGIN = pd.Timestamp("2024-01-01")
FORECAST_START, FORECAST_END = pd.Timestamp("2026-03-30"), pd.Timestamp("2026-06-07")
PAYDAY_CAP = 14  # days_to/from payday are capped; also fills the start of history
EXPECTED_OPERATING_DAYS = {16: 4, 18: 5, 22: 6}  # user reference, 2026 ISO weeks


def payday_rule(dates: pd.Series) -> pd.Series:
    """Paydays: the 25th and the last day of each month; if that falls on a
    Sunday, the payday moves back to Saturday."""
    days = []
    for month in pd.period_range(dates.min(), dates.max(), freq="M"):
        for d in (month.start_time + pd.Timedelta(days=24), month.end_time.normalize()):
            days.append(d - pd.Timedelta(days=1) if d.dayofweek == 6 else d)
    return dates.isin(days).astype(int)


def days_to_event(flag: pd.Series, direction: str) -> pd.Series:
    """Days to the next (direction='next') or since the previous ('prev') day where
    flag == 1, counting the day itself as 0. NaN when there is no such day."""
    idx = np.arange(len(flag))
    pos = pd.Series(np.where(flag.values == 1, idx, np.nan))
    if direction == "next":
        return pos.bfill().values - idx
    return idx - pos.ffill().values


cal = pd.read_csv(CALENDAR, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
assert (cal["date"].diff().dropna() == pd.Timedelta(days=1)).all(), "calendar has gaps"

# ---------------------------------------------------------------- payday
rule = payday_rule(cal["date"])
mismatch = cal.loc[rule != cal["is_payday"], ["date", "is_payday"]]
assert mismatch.empty, f"payday rule does not match calendar:\n{mismatch}"
print(f"Payday rule (25th and month-end, Sunday -> Saturday) matches all {int(cal['is_payday'].sum())} calendar paydays.")
cal["days_to_payday"] = days_to_event(cal["is_payday"], "next")
cal["days_since_payday"] = days_to_event(cal["is_payday"], "prev")
for c in ["days_to_payday", "days_since_payday"]:
    cal[c] = cal[c].fillna(PAYDAY_CAP).clip(upper=PAYDAY_CAP).astype(int)

# ---------------------------------------------------------------- festivals
cal["festival_ramp_sq"] = cal["festival_ramp"] ** 2
# Name of the festival the ramp leads up to: the next festival date on or after the day
fest_date = cal["date"].where(cal["festival"].notna())
fest_name = cal["festival"]
cal["next_festival"] = fest_name.bfill()
cal["days_to_festival"] = (fest_date.bfill() - cal["date"]).dt.days
cal["festival_name"] = np.where(cal["festival_ramp"] > 0, cal["next_festival"], "none")
# The ramp must belong to that festival: 0..9 days before it
ramp_days = cal.loc[cal["festival_ramp"] > 0, "days_to_festival"]
assert ramp_days.between(0, 9).all(), "festival_ramp > 0 outside the 9 days before a festival"
assert (cal.loc[cal["festival"].notna(), "festival_ramp"] == 1).all()
cal = cal.drop(columns=["next_festival"])

# ---------------------------------------------------------------- closures
op = cal["is_operating"]
prev_op_pos = pd.Series(np.where(op.shift(1).values == 1, np.arange(len(cal)) - 1, np.nan)).ffill()
cal["days_since_last_operating_day"] = np.arange(len(cal)) - prev_op_pos.values
# 2024-01-01 (Mon): the previous operating day is Sat 2023-12-30, before the calendar
cal.loc[0, "days_since_last_operating_day"] = 2 if cal.loc[0, "dow"] == 0 else 1
cal["days_since_last_operating_day"] = cal["days_since_last_operating_day"].astype(int)
# (Phase 4: the "first operating day after 2+ closed days" flag was dropped; only 2 events,
#  no catch-up effect. task2a/analysis.py derives it locally for the Phase 2 analysis.)
# First operating day after a single closed mid-week day (not the Monday after a Sunday):
# the day after May Day or Christmas (Phase 2: Fresh x1.12)
cal["after_midweek_closure"] = ((op == 1) & (cal["days_since_last_operating_day"] == 2)
                                & (cal["dow"] != 0)).astype(int)
# Closures other than a plain Sunday (festivals and public holidays). days_to_next_closure
# (any closure) was dropped in Phase 3: it is collinear with the weekday.
holiday_closed = ((op == 0) & (cal["dow"] != 6)).astype(int)
cal["days_to_next_holiday_closure"] = days_to_event(holiday_closed, "next").clip(max=PAYDAY_CAP)
cal["days_to_next_holiday_closure"] = cal["days_to_next_holiday_closure"].fillna(PAYDAY_CAP).astype(int)

# ---------------------------------------------------------------- payday window
# Phase 2: the Fresh payday lift lasts three days (payday and the next two days).
# Two ways to count the tail, compared in the Phase 3 backtest:
#   payday_cal_dK: K calendar days after the latest payday (the tail can land on a closed day)
#   payday_op_dK:  the K-th operating day after the latest payday (closed days are skipped)
for k in range(3):
    cal[f"payday_cal_d{k}"] = (cal["days_since_payday"] == k).astype(int)
op_count = op.cumsum()  # operating days up to and including each date
last_pay_opcount = op_count.where(cal["is_payday"] == 1).ffill()
op_since_payday = op_count - last_pay_opcount  # operating days after the latest payday
recent = cal["days_since_payday"] <= 7
cal["payday_op_d0"] = ((cal["is_payday"] == 1) & (op == 1)).astype(int)
for k in (1, 2):
    cal[f"payday_op_d{k}"] = ((op == 1) & recent & (op_since_payday == k)).astype(int)

# ---------------------------------------------------------------- other
cal["month"] = cal["date"].dt.month
cal["trend_years"] = (cal["date"] - TREND_ORIGIN).dt.days / 365.25
# Style spike weeks (Phase 2): first full week of March (W10), of August (W32), and W51
cal["style_peak_week"] = cal["iso_week"].isin([10, 32, 51]).astype(int)

PAYDAY_FLAGS = [f"payday_{v}_d{k}" for v in ("cal", "op") for k in range(3)]
calendar_features = [
    "dow", "is_operating", "is_payday", "days_to_payday", "days_since_payday", *PAYDAY_FLAGS,
    "festival_ramp", "festival_ramp_sq", "festival_name", "is_holiday", "monsoon",
    "month", "trend_years", "days_since_last_operating_day",
    "after_midweek_closure", "days_to_next_holiday_closure",
    "style_peak_week",
]
cal_out = cal[["date", "iso_year", "iso_week", "festival"] + calendar_features]

# ---------------------------------------------------------------- series x date
outlets = pd.read_csv(OUTLETS)
n_outlets = outlets.groupby(["depot", "brand"]).size().rename("n_outlets").reset_index()
labels = pd.read_csv(DATA / "daily_volume.csv", parse_dates=["date"])
hist_end = labels["date"].max()

window = cal_out[(cal_out["date"] >= labels["date"].min()) & (cal_out["date"] <= FORECAST_END)]
daily = window.merge(n_outlets, how="cross")
daily = daily.merge(labels[["date", "depot", "brand", "total_volume", "chilled_volume", "n_orders"]],
                    on=["date", "depot", "brand"], how="left")
daily["split"] = np.where(daily["date"] <= hist_end, "history", "forecast")
daily = daily[["date", "depot", "brand", "split", "iso_year", "iso_week", "festival", "n_outlets"]
              + calendar_features + ["total_volume", "chilled_volume", "n_orders"]]
daily = daily.sort_values(["depot", "brand", "date"]).reset_index(drop=True)

# ---------------------------------------------------------------- checks
hist = daily[daily["split"] == "history"]
fc = daily[daily["split"] == "forecast"]
assert hist["total_volume"].notna().all(), "history row without a label"
assert len(hist) == len(labels)
assert fc["date"].min() == FORECAST_START and fc["date"].max() == FORECAST_END
assert fc["date"].min() == hist_end + pd.Timedelta(days=1), "gap between history and forecast"
feature_cols = ["n_outlets"] + calendar_features
assert fc[feature_cols].notna().all().all(), "missing feature in the forecast window"
assert hist[feature_cols].notna().all().all(), "missing feature in history"
fc_weeks = sorted(set(zip(fc["iso_year"], fc["iso_week"])))
assert fc_weeks == [(2026, w) for w in range(14, 24)], fc_weeks
inputs = pd.read_csv(TASK2A_INPUTS)
assert set(zip(inputs["iso_year"], inputs["iso_week"])) == set(fc_weeks)
assert set(zip(inputs["depot"], inputs["brand"])) == set(zip(fc["depot"], fc["brand"]))
assert (fc.groupby(["depot", "brand", "iso_week"]).size() == 7).all()

op_days = fc.drop_duplicates("date").groupby("iso_week")["is_operating"].sum()
for wk, n in EXPECTED_OPERATING_DAYS.items():
    assert op_days[wk] == n, f"2026-W{wk}: {op_days[wk]} operating days, expected {n}"

print(f"History: {hist['date'].min().date()} to {hist['date'].max().date()} ({len(hist):,} rows)")
print(f"Forecast: {fc['date'].min().date()} to {fc['date'].max().date()} ({len(fc):,} rows), "
      f"ISO weeks 2026-W{fc_weeks[0][1]}..W{fc_weeks[-1][1]}, no missing features")
print("Operating days per forecast week:", op_days.to_dict())
print("Forecast-window festival days:")
print(cal_out[(cal_out["date"] >= FORECAST_START) & (cal_out["date"] <= FORECAST_END)
              & ((cal_out["is_holiday"] == 1) | (cal_out["is_operating"] == 0) & (cal_out["dow"] != 6))]
      [["date", "iso_week", "dow", "festival", "is_holiday", "is_operating"]].to_string(index=False))
print("Payday flags around 2026-W22/W23 (May 25 payday, May 30 payday moved from Sun May 31, Poson May 30):")
around = cal_out[(cal_out["date"] >= "2026-05-23") & (cal_out["date"] <= "2026-06-03")]
print(around[["date", "iso_week", "dow", "is_operating", "festival"] + PAYDAY_FLAGS].to_string(index=False))
w23 = around[around["iso_week"] == 23]
assert w23["payday_cal_d2"].sum() == 1 and w23["payday_op_d1"].sum() == 1 and w23["payday_op_d2"].sum() == 1, \
    "payday tail must spill into 2026-W23"
print("n_outlets:", n_outlets.set_index(["depot", "brand"])["n_outlets"].to_dict())

# ---------------------------------------------------------------- Style outlet x weekday (Phase 4)
# Every Style outlet orders once a week on one fixed weekday; when that day is closed the
# order is skipped (not moved). One row per outlet per date of its weekday.
orders = pd.concat([pd.read_csv(f, usecols=["delivery_id", "order_date", "outlet_id", "brand", "depot",
                                            "order_volume_m3"]) for f in ORDER_FILES])
style = orders[orders["brand"] == "Style"].copy()
style["date"] = pd.to_datetime(style["order_date"])
style["dow"] = style["date"].dt.dayofweek
outlet_dow = style.groupby("outlet_id")["dow"].agg(lambda x: sorted(set(x)))
assert (outlet_dow.str.len() == 1).all(), "a Style outlet orders on more than one weekday"
outlet_dow = outlet_dow.str[0].rename("outlet_dow")
style_outlets = outlets[outlets["brand"] == "Style"][["outlet_id", "depot"]].merge(outlet_dow, on="outlet_id")
assert len(style_outlets) == (outlets["brand"] == "Style").sum()
assert not style.duplicated(["outlet_id", "date"]).any()
so = window.merge(style_outlets, left_on="dow", right_on="outlet_dow").drop(columns="outlet_dow")
so = so.merge(style[["outlet_id", "date", "order_volume_m3"]], on=["outlet_id", "date"], how="left")
so["split"] = np.where(so["date"] <= hist_end, "history", "forecast")
so_hist = so[so["split"] == "history"]
assert so_hist.loc[so_hist["is_operating"] == 0, "order_volume_m3"].isna().all()
missing = so_hist[(so_hist["is_operating"] == 1) & so_hist["order_volume_m3"].isna()]
assert missing.empty, f"Style outlet without an order on an operating day of its weekday:\n{missing.head()}"
assert so_hist["order_volume_m3"].notna().sum() == len(style)
so["brand"] = "Style"
so = so.sort_values(["outlet_id", "date"]).reset_index(drop=True)
print("Style outlets: one order per operating day of the outlet's weekday, none on closed days "
      f"({len(style)} orders, {len(style_outlets)} outlets); weekday counts:",
      style_outlets.groupby(["depot", "outlet_dow"]).size().to_dict())

DATA.mkdir(parents=True, exist_ok=True)
cal_out.to_csv(DATA / "calendar_features.csv", index=False)
daily.to_csv(DATA / "daily_features.csv", index=False)
so.to_csv(DATA / "style_outlet_days.csv", index=False)
print(f"Saved: {DATA / 'calendar_features.csv'} ({len(cal_out)} rows), {DATA / 'daily_features.csv'} ({len(daily):,} rows), "
      f"{DATA / 'style_outlet_days.csv'} ({len(so):,} rows)")
