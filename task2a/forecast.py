"""Turn a model specification into weekly forecasts for the 6 depot x brand series.

Fitting and predicting are separate (`fit_spec` / `predict_spec`, `fit_config` /
`predict_config`), so the final model can be fitted once, saved, and used for
inference without retraining. The backtest (`Forecaster`) runs the same two
functions, so the submission is produced by exactly the code that was backtested.

Specs are small dicts, e.g.
    {"model": "glm", "include_midweek": False}
    {"model": "hgb", "max_depth": 4, "learning_rate": 0.06, "max_iter": 600}
    {"model": "style_outlet"}
    {"model": "tech_smooth", "kind": "ewm", "param": 13}
    {"model": "tech_calendar", "kind": "mean", "param": 52}
    {"model": "blend", "parts": [[spec_a, 0.6], [spec_b, 0.4]]}
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

from models import (BRANDS, SERIES, WEEK_KEYS, GlobalHGB, SeriesGLM, StyleOutletGLM, chilled_share_table,
                    smooth_last, week_capacity, weekday_weights)


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


def _params(spec):
    return {k: v for k, v in spec.items() if k != "model"}


def _series(df, depot, brand):
    return df[(df["depot"] == depot) & (df["brand"] == brand)]


# ---------------------------------------------------------------- fit
def fit_spec(spec, ctx: Context, brands=BRANDS, target: str = "total_volume") -> dict:
    """Fit one spec on the training part of ctx. Returns a picklable dict."""
    m, kw = spec["model"], _params(spec)
    fitted = {"spec": spec, "share": chilled_share_table(ctx.daily_train)}
    if m == "glm":
        fitted["models"] = {(d, b): SeriesGLM(b, target, **kw).fit(_series(ctx.daily_train, d, b))
                            for d, b in SERIES if b in brands}
    elif m == "hgb":
        fitted["model"] = GlobalHGB(**kw).fit(ctx.daily_train, target)
        fitted["brands"] = list(brands)
    elif m == "style_outlet":
        fitted["model"] = StyleOutletGLM(**kw).fit(ctx.outlet_train)
    elif m in ("tech_smooth", "tech_calendar"):
        fitted["depots"] = {}
        for depot in sorted(ctx.daily_train["depot"].unique()):
            wtr = _series(ctx.weekly_train, depot, "Tech").sort_values(["iso_year", "iso_week"])
            if m == "tech_smooth":
                fitted["depots"][depot] = {"level": smooth_last(wtr[target].to_numpy(), kw["kind"], kw["param"])}
            else:
                # level per unit of weekday capacity; forecast = level x the week's capacity
                weights = weekday_weights(_series(ctx.daily_train, depot, "Tech"), target)
                cap = week_capacity(_series(ctx.daily_train, depot, "Tech"), weights)
                wk = wtr.merge(cap, on=["iso_year", "iso_week"])
                wk = wk[wk["cap"] > 0]
                fitted["depots"][depot] = {"weights": weights,
                                           "level": smooth_last((wk[target] / wk["cap"]).to_numpy(), kw["kind"], kw["param"])}
    elif m == "blend":
        fitted["parts"] = [(fit_spec(part, ctx, brands, target), w) for part, w in spec["parts"]]
    else:
        raise ValueError(m)
    return fitted


# ---------------------------------------------------------------- predict
def _weekly_from_daily(daily_pred: pd.DataFrame, share: pd.Series) -> pd.DataFrame:
    """Sum daily predictions to weeks; chilled = Fresh daily total x depot-month chilled share."""
    d = daily_pred.copy()
    s = pd.Series([share.get((dep, mo), 0.0) for dep, mo in zip(d["depot"], d["month"])], index=d.index)
    d["chilled"] = np.where(d["brand"] == "Fresh", d["total"] * s, 0.0)
    return d.groupby(WEEK_KEYS, as_index=False)[["total", "chilled"]].sum()


DAY_COLS = ["depot", "brand", "date", "month", "iso_year", "iso_week"]


def predict_spec(fitted: dict, ctx: Context) -> pd.DataFrame:
    """Weekly forecasts (WEEK_KEYS + total + chilled) for the test part of ctx."""
    m = fitted["spec"]["model"]
    if m == "glm":
        rows = []
        for (d, b), model in fitted["models"].items():
            te = _series(ctx.daily_test, d, b)
            rows.append(te[DAY_COLS].assign(total=model.predict(te)))
        out = _weekly_from_daily(pd.concat(rows), fitted["share"])
    elif m == "hgb":
        te = ctx.daily_test[ctx.daily_test["brand"].isin(fitted["brands"])]
        out = _weekly_from_daily(te[DAY_COLS].assign(total=fitted["model"].predict(te)), fitted["share"])
    elif m == "style_outlet":
        te = ctx.outlet_test
        d = te[DAY_COLS].assign(total=fitted["model"].predict(te)).groupby(DAY_COLS, as_index=False)["total"].sum()
        tw = ctx.target_weeks[ctx.target_weeks["brand"] == "Style"]
        out = tw.merge(_weekly_from_daily(d, fitted["share"]), on=WEEK_KEYS, how="left").fillna(
            {"total": 0.0, "chilled": 0.0})
    elif m in ("tech_smooth", "tech_calendar"):
        rows = []
        for depot, par in fitted["depots"].items():
            tw = _series(ctx.target_weeks, depot, "Tech")
            if m == "tech_smooth":
                rows.append(tw.assign(total=par["level"]))
            else:
                cap = week_capacity(_series(ctx.daily_test, depot, "Tech"), par["weights"])
                p = tw.merge(cap, on=["iso_year", "iso_week"], how="left")
                assert p["cap"].notna().all()
                rows.append(p.assign(total=par["level"] * p["cap"]).drop(columns="cap"))
        out = pd.concat(rows).assign(chilled=0.0)[WEEK_KEYS + ["total", "chilled"]]
    elif m == "blend":
        out = None
        for part, w in fitted["parts"]:
            p = predict_spec(part, ctx).set_index(WEEK_KEYS)[["total", "chilled"]] * w
            out = p if out is None else out.add(p)
        out = out.dropna().reset_index()  # rows covered by every part
    else:
        raise ValueError(m)
    return out.sort_values(WEEK_KEYS).reset_index(drop=True)


# ---------------------------------------------------------------- configs
def assemble(config, by_brand: dict, chilled_direct, target_weeks) -> pd.DataFrame:
    """Combine per-brand weekly forecasts and apply the output rules:
    total >= 0, chilled only for Fresh, 0 <= chilled <= total."""
    out = pd.concat([by_brand[b][by_brand[b]["brand"] == b] for b in BRANDS], ignore_index=True)
    if config["chilled"]["method"] == "direct":
        direct = chilled_direct.set_index(WEEK_KEYS)["chilled"]
        out["chilled"] = np.where(out["brand"] == "Fresh",
                                  direct.reindex(out.set_index(WEEK_KEYS).index).to_numpy(), 0.0)
    out["chilled"] = np.where(out["brand"] == "Fresh", out["chilled"], 0.0)
    out["total"] = out["total"].clip(lower=0.0)
    out["chilled"] = np.minimum(out["chilled"].clip(lower=0.0), out["total"])
    assert len(out) == len(target_weeks) and out.merge(target_weeks, on=WEEK_KEYS).shape[0] == len(target_weeks)
    assert out[["total", "chilled"]].notna().all().all()
    return out


def _chilled_glm_spec(config):
    return {"model": "glm", **config["chilled"].get("glm", {})}


def fit_config(config, ctx: Context) -> dict:
    fitted = {"config": config, "brands": {b: fit_spec(config[b], ctx, [b]) for b in BRANDS}, "chilled": None}
    if config["chilled"]["method"] == "direct":
        fitted["chilled"] = fit_spec(_chilled_glm_spec(config), ctx, ["Fresh"], target="chilled_volume")
    return fitted


def predict_config(fitted: dict, ctx: Context) -> pd.DataFrame:
    by_brand = {b: predict_spec(f, ctx) for b, f in fitted["brands"].items()}
    direct = None
    if fitted["chilled"] is not None:
        direct = predict_spec(fitted["chilled"], ctx).drop(columns="chilled").rename(columns={"total": "chilled"})
    return assemble(fitted["config"], by_brand, direct, ctx.target_weeks)


class Forecaster:
    """Backtest helper: runs specs on one Context, caching each spec's weekly output."""

    def __init__(self, ctx: Context):
        self.ctx, self.cache = ctx, {}

    def run(self, spec) -> pd.DataFrame:
        key = spec_key(spec)
        if key not in self.cache:
            if spec["model"] == "blend":  # reuse cached parts
                out = None
                for part, w in spec["parts"]:
                    p = self.run(part).set_index(WEEK_KEYS)[["total", "chilled"]] * w
                    out = p if out is None else out.add(p)
                out = out.dropna().reset_index().sort_values(WEEK_KEYS).reset_index(drop=True)
            else:
                out = predict_spec(fit_spec(spec, self.ctx), self.ctx)
            self.cache[key] = out
        return self.cache[key]

    def chilled_direct(self, config) -> pd.DataFrame:
        spec = _chilled_glm_spec(config)
        key = "chilled_direct|" + spec_key(spec)
        if key not in self.cache:
            p = predict_spec(fit_spec(spec, self.ctx, ["Fresh"], target="chilled_volume"), self.ctx)
            self.cache[key] = p.drop(columns="chilled").rename(columns={"total": "chilled"})
        return self.cache[key]

    def run_config(self, config) -> pd.DataFrame:
        by_brand = {b: self.run(config[b]) for b in BRANDS}
        direct = self.chilled_direct(config) if config["chilled"]["method"] == "direct" else None
        return assemble(config, by_brand, direct, self.ctx.target_weeks)
