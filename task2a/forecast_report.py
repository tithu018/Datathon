"""Task 2A forecast sanity report: all 60 predictions by series with operating days,
paydays and festival notes, compared with the last-8-weeks mean and the matching 2025
week; flags weeks more than 25% away from the last-8 mean. Also plots history + forecast.

Run from the repo root after task2a/make_submission.py. Writes
task2a/reports/final_forecast_table.csv and task2a/reports/final_forecast.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("task2a") / "data"
REPORTS = Path("task2a") / "reports"
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FLAG = 0.25
MATCH_FESTIVALS = ["new_year", "vesak", "poson"]  # festivals in the 2026 forecast window

sub = pd.read_csv(Path("outputs") / "submission_task2a.csv")
inputs = pd.read_csv(Path("Test Data") / "task2a_test_inputs.csv")
cal = pd.read_csv(DATA / "calendar_features.csv", parse_dates=["date"])
weekly = pd.read_csv(DATA / "weekly_volume.csv", parse_dates=["week_start"])
daily = pd.read_csv(DATA / "daily_features.csv", parse_dates=["date"])
outlet_days = pd.read_csv(DATA / "style_outlet_days.csv", parse_dates=["date"])
fc = inputs.merge(sub, on="row_id")

# ---------------------------------------------------------------- week context
fest = cal[cal["festival"].notna()]
fest_date = {(int(r.date.year), r.festival): r.date for r in fest.itertuples()}


def iso_week_of(ts):
    return int(ts.isocalendar().week)


def match_2025(year, week):
    """Matching 2025 week: same position relative to a forecast-window festival
    (the week before / of / after it), else the same ISO week."""
    for f in MATCH_FESTIVALS:
        rel = week - iso_week_of(fest_date[(year, f)])
        if -2 <= rel <= 1:
            w25 = iso_week_of(fest_date[(2025, f)]) + rel
            where = {-2: "2 weeks before", -1: "week before", 0: "week of", 1: "week after"}[rel]
            return w25, f"{where} {f}"
    return week, "same ISO week"


notes = []
for (y, w), g in cal.groupby(["iso_year", "iso_week"]):
    if not (y == 2026 and 14 <= w <= 23):
        continue
    closed = g[(g["is_operating"] == 0) & (g["dow"] != 6)]
    ramp = g[(g["festival_ramp"] > 0) & (g["is_operating"] == 1)]  # build-up on days that deliver
    n = []
    if len(closed):
        n.append("closed " + ", ".join(f"{DOW[d]} {t:%d %b}" + (f" ({f})" if isinstance(f, str) else "")
                                         for d, t, f in zip(closed["dow"], closed["date"], closed["festival"])))
    for f, r in ramp.groupby("festival_name"):
        n.append(f"{f} ramp up to {r['festival_ramp'].max():.1f}")
    fest_open = g[g["festival"].notna() & (g["is_operating"] == 1)]
    n += [f"{f} {t:%a %d %b} (open)" for f, t in zip(fest_open["festival"], fest_open["date"])]
    if g["style_peak_week"].max():
        n.append("Style peak week")
    m25, how = match_2025(y, w)
    notes.append({"iso_year": y, "iso_week": w, "week_start": g["date"].min(), "op_days": int(g["is_operating"].sum()),
                  "paydays": int(g["is_payday"].sum()),
                  "payday_tail": int(g[["payday_cal_d1", "payday_cal_d2"]].to_numpy().sum()),
                  "notes": "; ".join(n), "match_2025_week": m25, "match_rule": how})
notes = pd.DataFrame(notes)
fc = fc.merge(notes, on=["iso_year", "iso_week"])

# last-8-weeks mean and the matched 2025 week, per series
hist_end = weekly.sort_values(["iso_year", "iso_week"]).groupby(["depot", "brand"]).tail(8)
last8 = hist_end.groupby(["depot", "brand"])["total_volume"].mean().rename("last8_mean")
fc = fc.merge(last8.reset_index(), on=["depot", "brand"])
w25 = weekly[weekly["iso_year"] == 2025].set_index(["depot", "brand", "iso_week"])
fc["actual_2025_match"] = [w25.loc[(d, b, w), "total_volume"] for d, b, w in zip(fc["depot"], fc["brand"], fc["match_2025_week"])]
fc["op_days_2025_match"] = [int(w25.loc[(d, b, w), "operating_days"]) for d, b, w in zip(fc["depot"], fc["brand"], fc["match_2025_week"])]
fc["vs_last8"] = fc["pred_total_volume_m3"] / fc["last8_mean"] - 1
fc["vs_2025_match"] = fc["pred_total_volume_m3"] / fc["actual_2025_match"] - 1


def lost_capacity(depot, brand, y, w):
    """Share of a normal week's expected volume lost to non-Sunday closures (weekday-weighted)."""
    d = daily[(daily["depot"] == depot) & (daily["brand"] == brand)]
    h = d[(d["split"] == "history") & (d["is_operating"] == 1)]
    weights = h.groupby("dow")["total_volume"].mean()
    week = d[(d["iso_year"] == y) & (d["iso_week"] == w)]
    full = weights.reindex(range(6)).fillna(0).sum()
    open_ = sum(weights.get(dw, 0) for dw, op in zip(week["dow"], week["is_operating"]) if op == 1)
    return 1 - open_ / full


def explain(r):
    parts = []
    lost = lost_capacity(r["depot"], r["brand"], r["iso_year"], r["iso_week"])
    if lost > 0.005:
        parts.append(f"closures remove {lost:.0%} of a normal week's {r['brand']} volume ({r['op_days']} operating days)")
    if r["brand"] == "Style":
        od = outlet_days[(outlet_days["depot"] == r["depot"]) & (outlet_days["iso_year"] == r["iso_year"])
                         & (outlet_days["iso_week"] == r["iso_week"])]
        skipped = int((od["is_operating"] == 0).sum())
        if skipped:
            parts.append(f"{skipped} of {len(od)} Style outlets skip their order (delivery day closed)")
    if "ramp" in r["notes"]:
        parts.append("festival build-up: " + "; ".join(p for p in r["notes"].split("; ") if "ramp" in p))
    if r["paydays"] >= 2:
        parts.append(f"{r['paydays']} paydays")
    if "Style peak" in r["notes"] and r["brand"] == "Style":
        parts.append("Style peak week")
    return "; ".join(parts) if parts else "no calendar driver found"


fc["flag"] = fc["vs_last8"].abs() > FLAG
fc["explanation"] = [explain(r) if f else "" for (_, r), f in zip(fc.iterrows(), fc["flag"])]
fc = fc.sort_values(["depot", "brand", "iso_week"])
cols = ["row_id", "depot", "brand", "iso_week", "op_days", "paydays", "payday_tail", "pred_total_volume_m3",
        "pred_chilled_volume_m3", "last8_mean", "vs_last8", "match_2025_week", "actual_2025_match", "vs_2025_match",
        "notes", "flag", "explanation"]
fc[cols].to_csv(REPORTS / "final_forecast_table.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)
print("Week context (2026):")
print(notes.assign(week_start=notes["week_start"].dt.strftime("%d %b")).to_string(index=False))
for (d, b), g in fc.groupby(["depot", "brand"], sort=False):
    t = g[["iso_week", "op_days", "paydays", "pred_total_volume_m3", "pred_chilled_volume_m3", "vs_last8",
           "match_2025_week", "actual_2025_match", "op_days_2025_match", "vs_2025_match", "flag"]].copy()
    t["vs_last8"] = t["vs_last8"].map("{:+.0%}".format)
    t["vs_2025_match"] = t["vs_2025_match"].map("{:+.0%}".format)
    t["flag"] = np.where(t["flag"], "<-- >25%", "")
    print(f"\n{d} {b}  (last-8-weeks mean {g['last8_mean'].iloc[0]:.1f} m3)")
    print(t.round(1).to_string(index=False))
print("\nFlagged weeks (|forecast / last-8 mean - 1| > 25%):")
for _, r in fc[fc["flag"]].iterrows():
    print(f"  {r['depot']} {r['brand']} W{r['iso_week']}: {r['pred_total_volume_m3']:.1f} vs last-8 {r['last8_mean']:.1f} "
          f"({r['vs_last8']:+.0%}); 2025 match W{r['match_2025_week']} {r['actual_2025_match']:.1f}. {r['explanation']}")

# ---------------------------------------------------------------- plot
fig, axes = plt.subplots(3, 2, figsize=(16, 11))
for ax, ((d, b), g) in zip(axes.T.flat, fc.groupby(["depot", "brand"], sort=False)):
    h = weekly[(weekly["depot"] == d) & (weekly["brand"] == b)]
    ax.plot(h["week_start"], h["total_volume"], color="0.4", lw=1, label="history (total)")
    ws = g["week_start"]
    ax.plot(ws, g["pred_total_volume_m3"], "o-", color="tab:red", lw=1.5, ms=3, label="forecast (total)")
    if b == "Fresh":
        ax.plot(h["week_start"], h["chilled_volume"], color="tab:blue", lw=0.8, alpha=0.6, label="history (chilled)")
        ax.plot(ws, g["pred_chilled_volume_m3"], "o-", color="tab:purple", lw=1.2, ms=3, label="forecast (chilled)")
    h25 = h[(h["iso_year"] == 2025) & h["iso_week"].between(14, 23)]
    ax.plot(h25["week_start"] + pd.Timedelta(weeks=52), h25["total_volume"], ":", color="tab:green", lw=1.2,
            label="2025 W14-23, shifted +52 weeks")
    ax.axvline(pd.Timestamp("2026-03-30"), color="k", lw=0.6, ls="--")
    ax.set_title(f"{d} {b}: weekly volume (m3)")
    ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(REPORTS / "final_forecast.png", dpi=120)
plt.close(fig)
print(f"\nSaved: {REPORTS / 'final_forecast_table.csv'}, {REPORTS / 'final_forecast.png'}")
