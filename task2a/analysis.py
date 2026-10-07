"""Task 2A exploratory analysis (Phase 2). Prints the key numbers and saves
plots plus task2a/reports/phase2_analysis.json.

Multipliers are measured on a de-trended scale: each operating day's volume is
divided by a local level, the centered 57-day mean of that series over "clean"
operating days (no festival ramp, holiday, payday or post-closure day). Weekday
effects come from clean days; payday, ramp and catch-up effects are further
divided by the brand's weekday multiplier.

Run from the repo root after task2a/prepare_features.py.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("task2a") / "data"
REPORTS = Path("task2a") / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
BRANDS = ["Fresh", "Style", "Tech"]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LEVEL_WINDOW = 57

df = pd.read_csv(DATA / "daily_features.csv", parse_dates=["date"])
# First operating day after 2+ consecutive closed days (no longer a model feature; Phase 4)
df["is_first_day_after_closure"] = ((df["is_operating"] == 1) & (df["days_since_last_operating_day"] >= 3)).astype(int)
cal = pd.read_csv(DATA / "calendar_features.csv", parse_dates=["date"])
hist = df[df["split"] == "history"].copy()
weekly = pd.read_csv(DATA / "weekly_volume.csv", parse_dates=["week_start"])
summary = {}


def add_level(d, value="total_volume"):
    """Local level per series from clean operating days, then normalized volume."""
    d = d.sort_values("date").copy()
    clean = ((d["is_operating"] == 1) & (d["festival_ramp"] == 0) & (d["is_holiday"] == 0)
             & (d["is_payday"] == 0) & (d["is_first_day_after_closure"] == 0))
    d["clean"] = clean
    d["level"] = d[value].where(clean).rolling(LEVEL_WINDOW, center=True, min_periods=10).mean()
    d["norm"] = np.where(d["is_operating"] == 1, d[value] / d["level"], np.nan)
    return d


hist = pd.concat([add_level(g) for _, g in hist.groupby(["depot", "brand"])])
op = hist[(hist["is_operating"] == 1) & hist["norm"].notna() & np.isfinite(hist["norm"])]

# ---------------------------------------------------------------- 1. weekday
print("1. Weekday multipliers (clean days, volume / local level)")
dow_mult = op[op["clean"]].groupby(["brand", "dow"])["norm"].mean().unstack("dow")
dow_mult.columns = [DOW[c] for c in dow_mult.columns]
print(dow_mult.round(3).to_string())
sat_vs_wed = dow_mult.loc["Fresh", "Sat"] / dow_mult.loc["Fresh", "Wed"] - 1
print(f"  Fresh Saturday vs Wednesday: {sat_vs_wed:+.1%}")
summary["weekday_multiplier"] = dow_mult.round(4).to_dict(orient="index")
summary["fresh_sat_vs_wed"] = round(float(sat_vs_wed), 4)

op = op.merge(dow_mult.stack().rename("dow_mult").reset_index().assign(
    dow=lambda x: x["level_1"].map(DOW.index)).drop(columns="level_1"), on=["brand", "dow"], how="left")
op["norm_adj"] = op["norm"] / op["dow_mult"]

# ---------------------------------------------------------------- 2. payday
print("\n2. Paydays")
pay = cal[cal["is_payday"] == 1].copy()
pay["dom"] = pay["date"].dt.day
pay["dow_name"] = pay["dow"].map(lambda i: DOW[i])
moved = pay[~pay["dom"].isin([25]) & (pay["date"] != pay["date"] + pd.offsets.MonthEnd(0))]
print(f"  {len(pay)} paydays {pay['date'].min().date()}..{pay['date'].max().date()}. Rule: the 25th and the "
      f"last day of each month; a Sunday payday moves back to Saturday ({len(moved)} moved).")
print("  Moved paydays:", ", ".join(f"{d.date()} ({w})" for d, w in zip(moved["date"], moved["dow_name"])))
closed_pay = pay[pay["is_operating"] == 0]
print("  Paydays on non-operating days:", ", ".join(str(d.date()) for d in closed_pay["date"]) or "none")
base = op[(op["festival_ramp"] == 0) & (op["is_first_day_after_closure"] == 0) & (op["is_holiday"] == 0)]
pay_eff = {}
for b in BRANDS:
    g = base[base["brand"] == b]
    on, off = g.loc[g["is_payday"] == 1, "norm_adj"], g.loc[g["is_payday"] == 0, "norm_adj"]
    # day after payday, to see if the effect carries over
    after = g.loc[g["days_since_payday"] == 1, "norm_adj"]
    pay_eff[b] = {"payday_ratio": round(float(on.mean() / off.mean()), 4),
                  "day_after_ratio": round(float(after.mean() / off.mean()), 4),
                  "n_payday_rows": int(len(on))}
    print(f"  {b:5s}: payday x{on.mean() / off.mean():.3f}, day after x{after.mean() / off.mean():.3f} "
          f"({len(on)} payday series-days)")
# Profile around paydays: does the effect last more than one day?
prof = {}
for k in range(-2, 4):
    col, val = ("days_since_payday", k) if k >= 0 else ("days_to_payday", -k)
    sel = base[(base[col] == val)]
    prof[k] = (sel.groupby("brand")["norm_adj"].mean() / base[base["is_payday"] == 0].groupby("brand")["norm_adj"].mean())
prof = pd.DataFrame(prof).round(3)
prof.columns = [f"{k:+d}" for k in prof.columns]
print("  Multiplier by day relative to payday (0 = payday; vs all non-payday days):")
print("  " + prof.to_string().replace("\n", "\n  "))
summary["payday_profile"] = prof.to_dict(orient="index")
summary["payday"] = {"rule": "25th and last day of month; Sunday -> preceding Saturday",
                     "n_paydays": int(len(pay)),
                     "moved_to_saturday": [str(d.date()) for d in moved["date"]],
                     "on_non_operating_days": [str(d.date()) for d in closed_pay["date"]],
                     "effect": pay_eff,
                     "dates": [str(d.date()) for d in pay["date"]]}

# ---------------------------------------------------------------- 3. festival ramp
print("\n3. Festival ramp multipliers (operating days, weekday-adjusted, payday excluded)")
r = op[(op["is_payday"] == 0) & (op["is_first_day_after_closure"] == 0)].copy()
r["ramp_bin"] = r["festival_ramp"].round(1)
ramp_mult = r.groupby(["brand", "ramp_bin"])["norm_adj"].mean().unstack("ramp_bin")
ramp_mult = ramp_mult.div(ramp_mult[0.0], axis=0)
print(ramp_mult.round(3).to_string())
summary["ramp_multiplier"] = {b: {str(k): round(float(v), 4) for k, v in row.items()}
                              for b, row in ramp_mult.iterrows()}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
dow_mult.T.plot(ax=axes[0], marker="o", title="Weekday multiplier (clean days)")
axes[0].axhline(1, color="grey", lw=0.8)
pd.DataFrame({b: [pay_eff[b]["payday_ratio"], pay_eff[b]["day_after_ratio"]] for b in BRANDS},
             index=["payday", "day after"]).plot.bar(ax=axes[1], title="Payday multiplier (weekday-adjusted)", rot=0)
axes[1].axhline(1, color="grey", lw=0.8)
ramp_mult.T.plot(ax=axes[2], marker="o", title="Festival-ramp multiplier (vs ramp = 0)")
axes[2].set_xlabel("festival_ramp")
axes[2].axhline(1, color="grey", lw=0.8)
fig.tight_layout()
fig.savefig(REPORTS / "multipliers.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- 4. festival dates
print("\n4. Festival dates by year")
fest = cal[cal["festival"].notna()].copy()
fest["year"] = fest["date"].dt.year
fest["label"] = fest.apply(lambda x: f"{x['date']:%m-%d} {DOW[x['dow']]} W{x['iso_week']}"
                           + ("" if x["is_operating"] else " closed"), axis=1)
fest_table = fest.pivot(index="festival", columns="year", values="label")
order = fest.groupby("festival")["date"].min().dt.dayofyear.sort_values().index
fest_table = fest_table.loc[order].fillna("-")
print(fest_table.to_string())
summary["festival_dates"] = fest_table.to_dict(orient="index")
extra_closed = cal[(cal["is_operating"] == 0) & (cal["dow"] != 6) & cal["festival"].isna()]
print("  Other non-Sunday closures:", ", ".join(f"{d.date()} ({DOW[w]})" for d, w in zip(extra_closed["date"], extra_closed["dow"])))
summary["other_closures"] = [str(d.date()) for d in extra_closed["date"]]

# ---------------------------------------------------------------- 5. festival-aligned Fresh
print("\n5. Daily Fresh volume aligned on festival dates (both depots, volume / local level)")
fresh = hist[hist["brand"] == "Fresh"].groupby("date").agg(
    total_volume=("total_volume", "sum"), **{c: (c, "first") for c in
    ["is_operating", "festival_ramp", "is_holiday", "is_payday", "is_first_day_after_closure", "dow"]}).reset_index()
fresh = add_level(fresh)
fresh = fresh.set_index("date")
aligned_rows, fest_stats = [], []
for _, f in fest.iterrows():
    if f["date"] > fresh.index.max():
        continue
    rng = pd.date_range(f["date"] - pd.Timedelta(days=10), f["date"] + pd.Timedelta(days=5))
    rng = rng[(rng >= fresh.index.min()) & (rng <= fresh.index.max())]
    seg = fresh.loc[rng]
    lvl = fresh.loc[f["date"] - pd.Timedelta(days=28):f["date"] - pd.Timedelta(days=11), "total_volume"]
    lvl = lvl[fresh.loc[lvl.index, "is_operating"] == 1].mean()  # pre-ramp baseline
    rel = seg["total_volume"] / lvl
    offs = (seg.index - f["date"]).days
    for o, v, isop in zip(offs, rel, seg["is_operating"]):
        aligned_rows.append({"festival": f["festival"], "year": f["year"], "offset": o, "ratio": v, "operating": isop})
    pre = rel[(offs >= -3) & (offs <= -1) & (seg["is_operating"] == 1).values]
    post = rel[(offs >= 1) & (offs <= 3) & (seg["is_operating"] == 1).values]
    fest_stats.append({"festival": f["festival"], "year": int(f["year"]),
                       "festival_day_operating": int(f["is_operating"]),
                       "peak_ratio_-10..-1": round(float(rel[(offs < 0)].max()), 3),
                       "mean_ratio_-3..-1": round(float(pre.mean()), 3),
                       "ratio_day0": round(float(rel[offs == 0].iloc[0]), 3),
                       "mean_ratio_+1..+3": round(float(post.mean()), 3)})
aligned = pd.DataFrame(aligned_rows)
fest_stats = pd.DataFrame(fest_stats)
print(fest_stats.to_string(index=False))
summary["festival_aligned_fresh"] = fest_stats.to_dict(orient="records")

fest_names = list(order)
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
for ax, name in zip(axes.flat, fest_names):
    for yr, g in aligned[aligned["festival"] == name].groupby("year"):
        ax.plot(g["offset"], g["ratio"], marker="o", ms=3, label=str(yr))
    ax.axvline(0, color="red", lw=0.8)
    ax.axhline(1, color="grey", lw=0.8)
    ax.set_title(name)
    ax.set_xlabel("days from festival")
    ax.legend(fontsize=8)
axes.flat[0].set_ylabel("Fresh volume / pre-ramp mean (0 = closed)")
for ax in list(axes.flat)[len(fest_names):]:
    ax.axis("off")
fig.suptitle("Daily Fresh volume (both depots) aligned on festival date")
fig.tight_layout()
fig.savefig(REPORTS / "festival_aligned_fresh.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- 6. catch-up after closures
print("\n6. Catch-up: first operating day after a 2+ day closure vs normal days of the same weekday")
catch = []
for b in BRANDS:
    g = op[op["brand"] == b]
    normal = g[g["clean"]]
    first = g[g["is_first_day_after_closure"] == 1]
    ratios = []
    for _, row in first.iterrows():
        ref = normal[(normal["dow"] == row["dow"]) & (normal["depot"] == row["depot"])]["norm"].mean()
        ratios.append(row["norm"] / ref)
    first = first.assign(ratio=ratios)
    ratios = [v for v in ratios if np.isfinite(v)]
    catch.append({"brand": b, "n_series_days": len(ratios),
                  "mean_ratio": round(float(np.mean(ratios)), 3),
                  "median_ratio": round(float(np.median(ratios)), 3)})
    if b == "Fresh":
        events = first.groupby("date")["ratio"].mean().round(3)
        print("  Fresh events:", ", ".join(f"{d.date()} {DOW[d.dayofweek]} x{v}" for d, v in events.items()))
catch = pd.DataFrame(catch)
print(catch.to_string(index=False))
# Single-day non-Sunday closures (Christmas, May Day): day after vs same weekday
one_day = op[(op["days_since_last_operating_day"] == 2) & (op["dow"] != 0)]
one = []
for b in BRANDS:
    g = one_day[one_day["brand"] == b]
    normal = op[(op["brand"] == b) & op["clean"]]
    r1 = [row["norm"] / normal[(normal["dow"] == row["dow"]) & (normal["depot"] == row["depot"])]["norm"].mean()
          for _, row in g.iterrows()]
    r1 = [v for v in r1 if np.isfinite(v)]  # skip weekdays with no scheduled orders (0/0)
    one.append({"brand": b, "n_series_days": len(r1), "mean_ratio": round(float(np.mean(r1)), 3)})
    if b == "Fresh":
        print("  Day after a 1-day mid-week closure, Fresh dates:",
              ", ".join(sorted({str(d.date()) for d in g["date"]})))
one = pd.DataFrame(one)
print("  Day after a 1-day mid-week closure vs same weekday:")
print("  " + one.to_string(index=False).replace("\n", "\n  "))
summary["after_one_day_closure"] = one.to_dict(orient="records")
summary["catch_up"] = catch.to_dict(orient="records")

# ---------------------------------------------------------------- 7. weekly series
fest_weeks = set(zip(fest["iso_year"], fest["iso_week"]))
fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True)
for (i, b), (j, d) in [((i, b), (j, d)) for i, b in enumerate(BRANDS) for j, d in enumerate(["Kandy", "Peliyagoda"])]:
    ax = axes[i, j]
    w = weekly[(weekly["depot"] == d) & (weekly["brand"] == b)]
    ax.plot(w["week_start"], w["total_volume"], lw=1)
    for _, fw in w[[k in fest_weeks for k in zip(w["iso_year"], w["iso_week"])]].iterrows():
        ax.axvline(fw["week_start"], color="red", alpha=0.35, lw=1)
    ax.set_title(f"{d} {b} weekly volume (m3); red = festival week")
fig.tight_layout()
fig.savefig(REPORTS / "weekly_series.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- 8. chilled share
print("\n8. Fresh chilled share (chilled / total)")
fw = weekly[weekly["brand"] == "Fresh"].copy()
fw["share"] = fw["chilled_volume"] / fw["total_volume"]
fw["month"] = fw["week_start"].dt.month
fw["year"] = fw["week_start"].dt.year
dm = hist[hist["brand"] == "Fresh"].assign(month=lambda x: x["date"].dt.month, year=lambda x: x["date"].dt.year)
share_month = dm.groupby(["year", "month"])[["chilled_volume", "total_volume"]].sum()
share_month = (share_month["chilled_volume"] / share_month["total_volume"]).unstack("year")
print(share_month.round(3).to_string())
by_depot = dm.groupby("depot")[["chilled_volume", "total_volume"]].sum()
print("  Overall by depot:", (by_depot["chilled_volume"] / by_depot["total_volume"]).round(4).to_dict())
summary["chilled_share_by_month"] = {str(c): {int(m): round(float(v), 4) for m, v in share_month[c].dropna().items()}
                                     for c in share_month.columns}
summary["chilled_share_by_depot"] = (by_depot["chilled_volume"] / by_depot["total_volume"]).round(4).to_dict()

fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
for d, g in fw.groupby("depot"):
    axes[0].plot(g["week_start"], g["share"], lw=1, label=d)
axes[0].set_title("Fresh weekly chilled share")
axes[0].legend()
share_month.plot(ax=axes[1], marker="o", title="Fresh chilled share by month")
axes[1].set_xticks(range(1, 13))
fig.tight_layout()
fig.savefig(REPORTS / "chilled_share.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- 9. Style seasonality
print("\n9. Style: festival weeks and month-by-month")
sw = weekly[weekly["brand"] == "Style"].copy()
sw["is_fest_week"] = [k in fest_weeks for k in zip(sw["iso_year"], sw["iso_week"])]
sw["per_op_day"] = sw["total_volume"] / sw["operating_days"]
style_fest = []
for d, g in sw.groupby("depot"):
    g = g.reset_index(drop=True)
    for i in g.index[g["is_fest_week"]]:
        nb = g.loc[[k for k in [i - 4, i - 3, i + 3, i + 4] if 0 <= k < len(g)], "per_op_day"].mean()
        pre = g.loc[[k for k in [i - 2, i - 1] if k >= 0], "per_op_day"].mean()
        style_fest.append({"depot": d, "week": f"{g.loc[i, 'iso_year']}-W{g.loc[i, 'iso_week']:02d}",
                           "fest_week_ratio": g.loc[i, "per_op_day"] / nb, "prev_2_weeks_ratio": pre / nb})
style_fest = pd.DataFrame(style_fest)
wk2fest = {(int(r.iso_year), int(r.iso_week)): r.festival for r in fest.itertuples()}
style_fest["festival"] = [wk2fest[(int(w[:4]), int(w[-2:]))] for w in style_fest["week"]]
by_fest = style_fest.groupby("festival")[["fest_week_ratio", "prev_2_weeks_ratio"]].mean().loc[
    [f for f in order if f in set(style_fest["festival"])]]
print("  Style by festival (festival week and 2 weeks before, vs weeks +-3..4 away):")
print("  " + by_fest.round(3).to_string().replace("\n", "\n  "))
print(f"  Volume per operating day vs weeks +-3..4 away: festival week x{style_fest['fest_week_ratio'].mean():.3f}, "
      f"2 weeks before x{style_fest['prev_2_weeks_ratio'].mean():.3f} (mean over {len(style_fest)} depot-festivals)")
# Spike weeks: volume per operating day > 1.25 x the centered 9-week median
spikes = []
fest_keys = sorted(wk2fest)
for d, g in sw.groupby("depot"):
    g = g.reset_index(drop=True)
    ratio = g["per_op_day"] / g["per_op_day"].rolling(9, center=True, min_periods=3).median()
    for i in g.index[ratio > 1.25]:
        key = (int(g.loc[i, "iso_year"]), int(g.loc[i, "iso_week"]))
        nxt = min([k for k in fest_keys if k >= key], default=None)
        weeks_to = None if nxt is None else (pd.Timestamp.fromisocalendar(*nxt, 1)
                                             - pd.Timestamp.fromisocalendar(*key, 1)).days // 7
        spikes.append({"depot": d, "week": f"{key[0]}-W{key[1]:02d}", "week_start": str(g.loc[i, "week_start"].date()),
                       "ratio": round(float(ratio[i]), 2), "festival_in_week": wk2fest.get(key, ""),
                       "next_festival": wk2fest.get(nxt, ""), "weeks_to_next_festival": weeks_to})
spikes = pd.DataFrame(spikes)
print("  Style spike weeks (per-operating-day volume > 1.25 x centered 9-week median):")
print("  " + spikes.to_string(index=False).replace("\n", "\n  "))
spike_weeks = spikes.assign(iso_week=spikes["week"].str[-2:].astype(int)).groupby("iso_week")["week"].apply(
    lambda s: sorted(set(x[:4] for x in s)))
print("  Spike ISO weeks and years:", spike_weeks.to_dict())
sd = hist[(hist["brand"] == "Style") & (hist["is_operating"] == 1)].assign(
    month=lambda x: x["date"].dt.month, year=lambda x: x["date"].dt.year)
style_month = sd.groupby(["year", "month"])["total_volume"].mean().unstack("year")
style_month_idx = style_month / style_month.mean()
print("  Style mean daily volume per operating day, by month (index vs that year's mean):")
print(style_month_idx.round(3).to_string())
style_dow = sd.groupby("dow")["total_volume"].mean()
print("  Style mean volume by weekday:", {DOW[k]: round(v, 2) for k, v in style_dow.items()})
style_dow_depot = sd.groupby(["depot", "dow"])["total_volume"].mean().unstack("dow").rename(columns=lambda c: DOW[c])
print("  Style mean volume by depot and weekday:")
print("  " + style_dow_depot.round(2).to_string().replace("\n", "\n  "))
summary["style"] = {"festival_week_ratio": round(float(style_fest["fest_week_ratio"].mean()), 4),
                    "pre_festival_2wk_ratio": round(float(style_fest["prev_2_weeks_ratio"].mean()), 4),
                    "by_festival": by_fest.round(4).to_dict(orient="index"),
                    "spike_weeks": spikes.to_dict(orient="records"),
                    "month_index": {str(c): {int(m): round(float(v), 4) for m, v in style_month_idx[c].dropna().items()}
                                    for c in style_month_idx.columns},
                    "weekday_mean_volume": {DOW[k]: round(float(v), 3) for k, v in style_dow.items()},
                    "weekday_mean_volume_by_depot": style_dow_depot.round(3).to_dict(orient="index")}

fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
for d, g in sw.groupby("depot"):
    axes[0].plot(g["week_start"], g["total_volume"], lw=1, label=d)
    fg = g[g["is_fest_week"]]
    axes[0].scatter(fg["week_start"], fg["total_volume"], color="red", zorder=3, s=18)
axes[0].set_title("Style weekly volume (red dots = festival weeks)")
axes[0].legend()
style_month_idx.plot(ax=axes[1], marker="o", title="Style volume per operating day by month (index)")
axes[1].axhline(1, color="grey", lw=0.8)
axes[1].set_xticks(range(1, 13))
fig.tight_layout()
fig.savefig(REPORTS / "style_seasonality.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- 10. growth
print("\n10. Year-on-year growth (mean volume per operating day, same months Jan-Mar)")
g = hist[(hist["is_operating"] == 1) & (hist["date"].dt.month <= 3)].assign(year=lambda x: x["date"].dt.year)
yoy = g.groupby(["brand", "year"])["total_volume"].mean().unstack("year")
yoy["2025_vs_2024"] = yoy[2025] / yoy[2024] - 1
yoy["2026_vs_2025"] = yoy[2026] / yoy[2025] - 1
print(yoy.round(3).to_string())
summary["yoy_growth_jan_mar"] = yoy[["2025_vs_2024", "2026_vs_2025"]].round(4).to_dict(orient="index")

(REPORTS / "phase2_analysis.json").write_text(json.dumps(summary, indent=2, default=str))
print("\nSaved plots: multipliers.png, festival_aligned_fresh.png, weekly_series.png, chilled_share.png, "
      "style_seasonality.png; summary: phase2_analysis.json (all in task2a/reports/)")
