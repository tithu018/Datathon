"""Turn a model specification into weekly forecasts for the 6 depot x brand series.

A Context holds the training history and the days/weeks to forecast. It is used
both by the backtest (Phase 4) and the final fit (Phase 5), so the submission is
produced by exactly the code that was backtested.

Specs are small dicts, e.g.
    {"model": "glm", "include_midweek": False}
    {"model": "hgb", "max_depth": 3, "learning_rate": 0.03, "max_iter": 600}
    {"model": "style_outlet", "halflife_weeks": 52}
    {"model": "tech_smooth", "kind": "ewm", "param": 13}
    {"model": "tech_calendar", "kind": "mean", "param": 26}
    {"model": "blend", "parts": [[spec_a, 0.7], [spec_b, 0.3]]}
A config maps each brand to a spec, plus the chilled method:
    {"Fresh": spec, "Style": spec, "Tech": spec,
     "chilled": {"method": "share" | "direct", "glm": {...}}}
Every spec returns weekly `total` and `chilled` (share-based for daily models,
0 for weekly Tech models); the config's chilled method decides which chilled is used.
"""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from models import (BRANDS, SERIES, WEEK_KEYS, GlobalHGB, SeriesGLM, StyleOutletGLM,
                    calendar_level_forecast, chilled_share_table, smooth_last)


@dataclass
class Context:
    daily_train: pd.DataFrame   # history, all series, labels + features
    daily_test: pd.DataFrame    # days to forecast, all series, features
    weekly_train: pd.DataFrame  # history, weekly labels
    target_weeks: pd.DataFrame  # WEEK_KEYS to forecast
    outlet_train: pd.DataFrame  # Style outlet-days, history
    outlet_test: pd.DataFrame   # Style outlet-days to forecast


def spec_key(spec) -> str:
    return json.dumps(spec, sort_keys=True)


def _weekly_from_daily(ctx: Context, daily_pred: pd.DataFrame) -> pd.DataFrame:
    """Sum daily predictions to weeks; chilled = Fresh daily total x depot-month chilled share."""
    share = chilled_share_table(ctx.daily_train)
    d = daily_pred.copy()
    s = pd.Series([share.get((dep, m), 0.0) for dep, m in zip(d["depot"], d["month"])], index=d.index)
    d["chilled"] = np.where(d["brand"] == "Fresh", d["total"] * s, 0.0)
    return d.groupby(WEEK_KEYS, as_index=False)[["total", "chilled"]].sum()


def _glm(ctx, spec, brands, target="total_volume"):
    kw = {k: v for k, v in spec.items() if k != "model"}
    rows = []
    for depot, brand in SERIES:
        if brand not in brands:
            continue
        tr = ctx.daily_train[(ctx.daily_train["depot"] == depot) & (ctx.daily_train["brand"] == brand)]
        te = ctx.daily_test[(ctx.daily_test["depot"] == depot) & (ctx.daily_test["brand"] == brand)]
        m = SeriesGLM(brand, target, **kw).fit(tr)
        rows.append(te[["depot", "brand", "date", "month", "iso_year", "iso_week"]].assign(total=m.predict(te)))
    return _weekly_from_daily(ctx, pd.concat(rows))


def _hgb(ctx, spec, brands):
    kw = {k: v for k, v in spec.items() if k != "model"}
    m = GlobalHGB(**kw).fit(ctx.daily_train)
    te = ctx.daily_test[ctx.daily_test["brand"].isin(brands)]
    d = te[["depot", "brand", "date", "month", "iso_year", "iso_week"]].assign(total=m.predict(te))
    return _weekly_from_daily(ctx, d)


def _style_outlet(ctx, spec):
    kw = {k: v for k, v in spec.items() if k != "model"}
    m = StyleOutletGLM(**kw).fit(ctx.outlet_train)
    te = ctx.outlet_test
    d = te[["depot", "brand", "date", "month", "iso_year", "iso_week"]].assign(total=m.predict(te))
    d = d.groupby(["depot", "brand", "date", "month", "iso_year", "iso_week"], as_index=False)["total"].sum()
    w = _weekly_from_daily(ctx, d)
    # weeks where no Style outlet's day is open still exist (with 0)
    tw = ctx.target_weeks[ctx.target_weeks["brand"] == "Style"]
    return tw.merge(w, on=WEEK_KEYS, how="left").fillna({"total": 0.0, "chilled": 0.0})


def _tech(ctx, spec):
    rows = []
    for depot in sorted(ctx.daily_train["depot"].unique()):
        wtr = ctx.weekly_train[(ctx.weekly_train["depot"] == depot) & (ctx.weekly_train["brand"] == "Tech")]
        wtr = wtr.sort_values(["iso_year", "iso_week"])
        tw = ctx.target_weeks[(ctx.target_weeks["depot"] == depot) & (ctx.target_weeks["brand"] == "Tech")]
        if spec["model"] == "tech_smooth":
            level = smooth_last(wtr["total_volume"].to_numpy(), spec["kind"], spec["param"])
            rows.append(tw.assign(total=level))
        else:
            dtr = ctx.daily_train[(ctx.daily_train["depot"] == depot) & (ctx.daily_train["brand"] == "Tech")]
            dte = ctx.daily_test[(ctx.daily_test["depot"] == depot) & (ctx.daily_test["brand"] == "Tech")]
            p = calendar_level_forecast(dtr, wtr, dte, spec["kind"], spec["param"])
            rows.append(tw.merge(p[["iso_year", "iso_week", "pred"]], on=["iso_year", "iso_week"]).rename(
                columns={"pred": "total"}))
    return pd.concat(rows).assign(chilled=0.0)[WEEK_KEYS + ["total", "chilled"]]


class Forecaster:
    """Runs specs on one Context, caching each spec's weekly output (all brands it supports)."""

    def __init__(self, ctx: Context):
        self.ctx, self.cache = ctx, {}

    def run(self, spec) -> pd.DataFrame:
        key = spec_key(spec)
        if key not in self.cache:
            m = spec["model"]
            if m == "glm":
                out = _glm(self.ctx, spec, BRANDS)
            elif m == "hgb":
                out = _hgb(self.ctx, spec, BRANDS)
            elif m == "style_outlet":
                out = _style_outlet(self.ctx, spec)
            elif m in ("tech_smooth", "tech_calendar"):
                out = _tech(self.ctx, spec)
            elif m == "blend":
                out = None
                for part, w in spec["parts"]:
                    p = self.run(part).set_index(WEEK_KEYS)[["total", "chilled"]] * w
                    out = p if out is None else out.add(p, fill_value=None)
                out = out.dropna().reset_index()
            else:
                raise ValueError(m)
            self.cache[key] = out.sort_values(WEEK_KEYS).reset_index(drop=True)
        return self.cache[key]

    def chilled_direct(self, glm_kwargs) -> pd.DataFrame:
        """Fresh chilled volume from a GLM fitted on chilled volume directly."""
        key = "chilled_direct|" + spec_key(glm_kwargs)
        if key not in self.cache:
            spec = {"model": "glm", **glm_kwargs}
            out = _glm(self.ctx, spec, ["Fresh"], target="chilled_volume")
            self.cache[key] = out[WEEK_KEYS + ["total"]].rename(columns={"total": "chilled"})
        return self.cache[key]

    def run_config(self, config) -> pd.DataFrame:
        parts = [self.run(config[b]).query("brand == @b") for b in BRANDS]
        out = pd.concat(parts, ignore_index=True)
        if config["chilled"]["method"] == "direct":
            direct = self.chilled_direct(config["chilled"]["glm"]).set_index(WEEK_KEYS)["chilled"]
            idx = out.set_index(WEEK_KEYS).index
            out["chilled"] = np.where(out["brand"] == "Fresh", direct.reindex(idx).to_numpy(), 0.0)
        out["chilled"] = np.where(out["brand"] == "Fresh", out["chilled"], 0.0)
        out["total"] = out["total"].clip(lower=0.0)
        out["chilled"] = np.minimum(out["chilled"].clip(lower=0.0), out["total"])
        tw = self.ctx.target_weeks
        assert len(out) == len(tw) and out.merge(tw, on=WEEK_KEYS).shape[0] == len(tw)
        return out
