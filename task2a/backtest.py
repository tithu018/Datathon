"""Task 2A rolling backtest (Phase 3): weekly baselines vs the daily Poisson GLM.

For each test period, every model is trained only on history before the
period's first week and forecasts its 10 weeks. Metrics: WAPE (sum|error| /
sum actual) and MAE of weekly volume, for total volume (all series) and chilled
volume (Fresh series), overall, per brand and per series.

Run from the repo root after task2a/prepare_features.py. Writes
task2a/reports/backtest_results.csv, backtest_metrics.csv,
backtest_glm_multipliers.csv, backtest_headline.png and backtest_output.txt.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import FESTIVALS, SeriesGLM, baseline_forecasts  # noqa: E402

DATA = Path("task2a") / "data"
REPORTS = Path("task2a") / "reports"
SERIES = [(d, b) for d in ["Kandy", "Peliyagoda"] for b in ["Fresh", "Style", "Tech"]]
PERIODS = {  # name: (iso_year, first week, last week)
    "2025-W14..W23 (headline)": (2025, 14, 23),
    "2026-W04..W13": (2026, 4, 13),
    "2025-W30..W39 (esala)": (2025, 30, 39),
    "2025-W40..W49 (deepavali)": (2025, 40, 49),
}
HEADLINE = "2025-W14..W23 (headline)"
GLM_MODELS = {"glm_cal": "cal", "glm_op": "op"}
BASELINES = ["last8_mean", "same_week_last_year", "mean13", "mean26", "ewm_hl8"]
TARGETS = {"total": "total_volume", "chilled": "chilled_volume"}

daily = pd.read_csv(DATA / "daily_features.csv", parse_dates=["date"])
daily = daily[daily["split"] == "history"].copy()
daily["wk"] = daily["iso_year"] * 100 + daily["iso_week"]
weekly = pd.read_csv(DATA / "weekly_volume.csv")
weekly["wk"] = weekly["iso_year"] * 100 + weekly["iso_week"]

rows = []
for period, (year, w0, w1) in PERIODS.items():
    start, end = year * 100 + w0, year * 100 + w1
    for depot, brand in SERIES:
        s_daily = daily[(daily["depot"] == depot) & (daily["brand"] == brand)]
        train, test = s_daily[s_daily["wk"] < start], s_daily[s_daily["wk"].between(start, end)]
        s_week = weekly[(weekly["depot"] == depot) & (weekly["brand"] == brand)].sort_values("wk")
        w_train, w_test = s_week[s_week["wk"] < start], s_week[s_week["wk"].between(start, end)]
        assert len(w_test) == 10 and train["date"].max() < test["date"].min()
        for tname, tcol in TARGETS.items():
            if tname == "chilled" and brand != "Fresh":
                continue
            preds = w_test[["iso_year", "iso_week", tcol]].rename(columns={tcol: "actual"}).reset_index(drop=True)
            for mname, payday in GLM_MODELS.items():
                glm = SeriesGLM(brand, tcol, payday).fit(train)
                p = test.assign(pred=glm.predict(test)).groupby(["iso_year", "iso_week"], as_index=False)["pred"].sum()
                preds[mname] = preds.merge(p, on=["iso_year", "iso_week"], how="left")["pred"].values
            base = baseline_forecasts(w_train, w_test, tcol).reset_index(drop=True)
            for b in BASELINES:
                preds[b] = base[b].values
            long = preds.melt(id_vars=["iso_year", "iso_week", "actual"], var_name="model", value_name="pred")
            long[["period", "depot", "brand", "target"]] = period, depot, brand, tname
            rows.append(long)

res = pd.concat(rows, ignore_index=True)
res = res[["period", "depot", "brand", "target", "iso_year", "iso_week", "model", "actual", "pred"]]
assert res["pred"].notna().all(), res[res["pred"].isna()].head()
res.to_csv(REPORTS / "backtest_results.csv", index=False)


# ---------------------------------------------------------------- metrics
def metrics(g):
    err = (g["pred"] - g["actual"]).abs()
    return pd.Series({"wape": err.sum() / g["actual"].sum(), "mae": err.mean(), "n_weeks": len(g)})


levels = []
for level, keys in {"overall": [], "brand": ["brand"], "series": ["depot", "brand"]}.items():
    m = res.groupby(["period", "target", "model"] + keys).apply(metrics, include_groups=False).reset_index()
    m["level"] = level
    if level == "overall":
        m["group"] = "all"
    elif level == "brand":
        m["group"] = m["brand"]
    else:
        m["group"] = m["depot"] + " " + m["brand"]
    levels.append(m[["period", "target", "level", "group", "model", "wape", "mae", "n_weeks"]])
met = pd.concat(levels, ignore_index=True)
met.to_csv(REPORTS / "backtest_metrics.csv", index=False)

MODELS = list(GLM_MODELS) + BASELINES
pd.set_option("display.width", 250)


def table(target, level, stat="wape", fmt="{:.1%}"):
    t = met[(met["target"] == target) & (met["level"] == level)]
    t = t.pivot_table(index=["group", "model"], columns="period", values=stat)[list(PERIODS)]
    t["mean"] = t.mean(axis=1)
    t = t.reindex([(g, mm) for g in t.index.get_level_values(0).unique() for mm in MODELS if (g, mm) in t.index])
    return t.map(fmt.format)


print("=== WAPE, total volume, overall ===")
print(table("total", "overall").droplevel(0).to_string())
print("\n=== WAPE, chilled volume (Fresh), overall ===")
print(table("chilled", "overall").droplevel(0).to_string())
print("\n=== WAPE, total volume, per brand ===")
print(table("total", "brand").to_string())
print("\n=== WAPE, total volume, per series (GLM payday=cal, last-8 mean, same week last year) ===")
ts = table("total", "series")
print(ts[ts.index.get_level_values(1).isin(["glm_cal", "last8_mean", "same_week_last_year"])].to_string())
print("\n=== MAE (m3/week), total volume, per series ===")
ms = table("total", "series", "mae", "{:.1f}")
print(ms[ms.index.get_level_values(1).isin(["glm_cal", "last8_mean", "same_week_last_year"])].to_string())
print("\n=== MAE (m3/week), chilled volume, per series ===")
mc = table("chilled", "series", "mae", "{:.1f}")
print(mc[mc.index.get_level_values(1).isin(["glm_cal", "glm_op", "last8_mean", "same_week_last_year"])].to_string())

# ---------------------------------------------------------------- payday counting
o = met[(met["level"] == "overall") & met["model"].isin(GLM_MODELS)]
pay_cmp = o.pivot_table(index=["target", "model"], columns="period", values="wape")[list(PERIODS)]
pay_cmp["mean"] = pay_cmp.mean(axis=1)
fresh_cmp = met[(met["level"] == "brand") & (met["group"] == "Fresh") & met["model"].isin(GLM_MODELS)
                & (met["target"] == "total")].pivot_table(index="model", columns="period", values="wape")[list(PERIODS)]
fresh_cmp["mean"] = fresh_cmp.mean(axis=1)
print("\n=== Payday tail: calendar days (glm_cal) vs operating days (glm_op), WAPE ===")
print(pay_cmp.map("{:.2%}".format).to_string())
print("Fresh total only:")
print(fresh_cmp.map("{:.2%}".format).to_string())
best_payday = "cal" if pay_cmp.loc[("total", "glm_cal"), "mean"] <= pay_cmp.loc[("total", "glm_op"), "mean"] else "op"
print(f"Chosen payday counting (lower mean total WAPE): {best_payday}")

# ---------------------------------------------------------------- multipliers vs Phase 2
print("\n=== GLM multipliers, Peliyagoda Fresh (fit on all history) vs Phase 2 empirical (Fresh, both depots) ===")
pel = daily[(daily["depot"] == "Peliyagoda") & (daily["brand"] == "Fresh")]
glm_full = SeriesGLM("Fresh", "total_volume", best_payday).fit(pel)
mult = glm_full.multipliers()
mult.rename("multiplier").to_csv(REPORTS / "backtest_glm_multipliers.csv")
p2 = json.loads((REPORTS / "phase2_analysis.json").read_text())
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
glm_dow = np.array([1.0] + [mult.get(f"dow_{d}", np.nan) for d in range(1, 6)])
emp_dow = np.array([p2["weekday_multiplier"]["Fresh"][d] for d in dow_names])
cmp_dow = pd.DataFrame({"GLM (vs Wed)": glm_dow / glm_dow[2], "Phase 2 (vs Wed)": emp_dow / emp_dow[2]}, index=dow_names)
print(cmp_dow.round(3).T.to_string())
prof = p2["payday_profile"]["Fresh"]
cmp_pay = pd.DataFrame({"GLM": [mult.get(f"payday_d{k}") for k in range(3)],
                        "Phase 2": [prof["+0"], prof["+1"], prof["+2"]]}, index=["d0", "d1", "d2"])
print(cmp_pay.round(3).T.to_string())
ramps = np.round(np.arange(0.1, 1.01, 0.1), 1)
b1, b2 = np.log(mult.get("ramp_new_year", 1.0)), np.log(mult.get("ramp2_new_year", 1.0))
cmp_ramp = pd.DataFrame({"GLM new_year": np.exp(b1 * ramps + b2 * ramps ** 2),
                         "Phase 2 all festivals": [p2["ramp_multiplier"]["Fresh"][str(r)] for r in ramps]},
                        index=ramps)
print(cmp_ramp.round(3).T.to_string())
print("  (Phase 2 festival-aligned New Year, both depots: days -3..-1 x1.63 (2024), x1.67 (2025); ramp 0.7-0.9)")
other = mult[[c for c in mult.index if c.startswith(("after_", "holiday_", "trend"))]]
print("  Other:", other.round(3).to_dict())

# ---------------------------------------------------------------- headline weekly tables
print("\n=== Headline 2025-W14..W23: weekly actual vs predicted (m3) ===")
for depot, brand in [("Peliyagoda", "Fresh"), ("Peliyagoda", "Style")]:
    h = res[(res["period"] == HEADLINE) & (res["depot"] == depot) & (res["brand"] == brand) & (res["target"] == "total")]
    t = h.pivot_table(index="iso_week", columns="model", values="pred")[["glm_cal", "glm_op", "last8_mean", "same_week_last_year"]]
    t.insert(0, "actual", h.drop_duplicates("iso_week").set_index("iso_week")["actual"])
    t["glm_cal_err%"] = (t["glm_cal"] / t["actual"] - 1) * 100
    print(f"{depot} {brand}:")
    print(t.round(1).to_string())

# ---------------------------------------------------------------- plot
fig, axes = plt.subplots(3, 2, figsize=(15, 11))
for ax, (depot, brand) in zip(axes.T.flat, SERIES):
    h = res[(res["period"] == HEADLINE) & (res["depot"] == depot) & (res["brand"] == brand) & (res["target"] == "total")]
    t = h.pivot_table(index="iso_week", columns="model", values="pred")
    act = h.drop_duplicates("iso_week").set_index("iso_week")["actual"]
    ax.plot(act.index, act.values, "k-o", label="actual", lw=2)
    ax.plot(t.index, t[f"glm_{best_payday}"], "-o", label=f"GLM (payday {best_payday})")
    ax.plot(t.index, t["last8_mean"], "--", label="last-8-weeks mean")
    ax.plot(t.index, t["same_week_last_year"], ":", label="same week last year")
    ax.set_title(f"{depot} {brand}, 2025 weeks 14-23")
    ax.set_xticks(range(14, 24))
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(REPORTS / "backtest_headline.png", dpi=120)
plt.close(fig)
print("\nSaved: task2a/reports/backtest_results.csv, backtest_metrics.csv, backtest_glm_multipliers.csv, backtest_headline.png")
