"""Task 2A evaluation: backtest periods, contexts and scoring (WAPE, MAE, signed bias)."""
from pathlib import Path

import pandas as pd

from forecast import Context
from models import WEEK_KEYS

DATA = Path("task2a") / "data"
SELECTION_PERIODS = {  # used for model selection
    "2025-W14..W23 (headline)": (2025, 14, 23),
    "2026-W04..W13": (2026, 4, 13),
    "2025-W30..W39 (esala)": (2025, 30, 39),
    "2025-W40..W49 (deepavali)": (2025, 40, 49),
}
HEADLINE = "2025-W14..W23 (headline)"
CONFIRMATION_PERIOD = {"2025-W02..W11 (confirmation)": (2025, 2, 11)}  # never used for selection


def load_history():
    daily = pd.read_csv(DATA / "daily_features.csv", parse_dates=["date"])
    daily = daily[daily["split"] == "history"].copy()
    weekly = pd.read_csv(DATA / "weekly_volume.csv")
    outlets = pd.read_csv(DATA / "style_outlet_days.csv", parse_dates=["date"])
    outlets = outlets[outlets["split"] == "history"].copy()
    for df in (daily, weekly, outlets):
        df["wk"] = df["iso_year"] * 100 + df["iso_week"]
    return daily, weekly, outlets


def make_context(daily, weekly, outlets, year, w0, w1):
    start, end = year * 100 + w0, year * 100 + w1
    test_w = weekly[weekly["wk"].between(start, end)]
    ctx = Context(
        daily_train=daily[daily["wk"] < start], daily_test=daily[daily["wk"].between(start, end)],
        weekly_train=weekly[weekly["wk"] < start], target_weeks=test_w[WEEK_KEYS].reset_index(drop=True),
        outlet_train=outlets[outlets["wk"] < start], outlet_test=outlets[outlets["wk"].between(start, end)])
    assert ctx.daily_train["date"].max() < ctx.daily_test["date"].min()
    assert len(ctx.target_weeks) == 60
    actual = test_w[WEEK_KEYS + ["total_volume", "chilled_volume"]].rename(
        columns={"total_volume": "actual_total", "chilled_volume": "actual_chilled"})
    return ctx, actual


def score(pred: pd.DataFrame, actual: pd.DataFrame, value: str = "total", headline: str = HEADLINE) -> dict:
    """pred: period + WEEK_KEYS + value; actual: period + WEEK_KEYS + actual_<value>.
    Returns WAPE/MAE/bias: mean over periods and the headline period, overall, per brand
    and per series. Chilled is scored on Fresh only."""
    df = pred.merge(actual, on=["period"] + WEEK_KEYS, how="inner")
    assert len(df) == len(pred), "prediction rows without actuals"
    if value == "chilled":
        df = df[df["brand"] == "Fresh"]
    df = df.assign(err=df[value] - df[f"actual_{value}"], abs_err=lambda x: x["err"].abs(),
                   act=df[f"actual_{value}"], series=df["depot"] + " " + df["brand"])

    def agg(keys):
        g = df.groupby(["period"] + keys)[["abs_err", "err", "act"]].sum()
        n = df.groupby(["period"] + keys).size()
        r = pd.DataFrame({"wape": g["abs_err"] / g["act"], "bias": g["err"] / g["act"], "mae": g["abs_err"] / n})
        return r

    out = {}
    for level, keys in {"overall": [], "brand": ["brand"], "series": ["series"]}.items():
        if level == "overall":
            g = df.groupby("period")[["abs_err", "err", "act"]].sum()
            n = df.groupby("period").size()
            r = pd.DataFrame({"wape": g["abs_err"] / g["act"], "bias": g["err"] / g["act"], "mae": g["abs_err"] / n})
            out["overall"] = {"mean": r.mean().to_dict(), "headline": r.loc[headline].to_dict() if headline in r.index else None}
        else:
            r = agg(keys)
            out[level] = {}
            for grp, sub in r.groupby(level=1):
                sub = sub.droplevel(1)
                out[level][grp] = {"mean": sub.mean().to_dict(),
                                   "headline": sub.loc[headline].to_dict() if headline in sub.index else None}
    out["per_period_overall"] = df.groupby("period").apply(
        lambda x: x["abs_err"].sum() / x["act"].sum(), include_groups=False).to_dict()
    return out
