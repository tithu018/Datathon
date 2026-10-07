"""Task 2A models.

- SeriesGLM: daily Poisson GLM per depot x brand series (Phase 3), with optional
  recency weighting / training window and the after_midweek_closure ablation.
- GlobalHGB: one HistGradientBoostingRegressor (Poisson loss) across all series.
- StyleOutletGLM: Style at outlet level. Each Style outlet orders once a week on a
  fixed weekday and skips the week when that day is closed, so weekly volume is
  the sum of predicted order sizes of the outlets whose day is open.
- Weekly smoothers and the calendar-shaped level for Tech.
- Chilled share table (depot x month) for the share-based chilled forecast.

All daily models are fitted on operating days; closed days are forced to 0.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor

SEED = 42
DEPOTS = ["Kandy", "Peliyagoda"]
BRANDS = ["Fresh", "Style", "Tech"]
SERIES = [(d, b) for d in DEPOTS for b in BRANDS]
FESTIVALS = ["thai_pongal", "new_year", "vesak", "poson", "esala", "deepavali", "christmas"]
GLM_ALPHA = 1e-4          # light L2 penalty: keeps rare dummies finite, leaves multipliers ~unpenalized
STRUCTURAL_ZERO_SHARE = 0.01  # a weekday below 1% of the series' mean daily volume is "no delivery"
WEEK_KEYS = ["depot", "brand", "iso_year", "iso_week"]


# ---------------------------------------------------------------- GLM
def design_matrix(df: pd.DataFrame, brand: str, payday: str = "cal", include_midweek: bool = True,
                  include_dow: bool = True) -> pd.DataFrame:
    """GLM features. `payday`: "cal" (calendar days) or "op" (operating days) tail counting."""
    X = pd.DataFrame(index=df.index)
    if include_dow:
        for d in range(1, 6):  # Monday is the reference weekday; Sunday is never operating
            X[f"dow_{d}"] = (df["dow"] == d).astype(float)
    for k in range(3):
        X[f"payday_d{k}"] = df[f"payday_{payday}_d{k}"].astype(float)
    for f in FESTIVALS:
        on = (df["festival_name"] == f).astype(float)
        X[f"ramp_{f}"] = df["festival_ramp"] * on
        X[f"ramp2_{f}"] = df["festival_ramp_sq"] * on
    if include_midweek:
        X["after_midweek_closure"] = df["after_midweek_closure"].astype(float)
    for k in (1, 2, 3):  # 1-3 days before a non-Sunday closure (demand pulled forward)
        X[f"holiday_closure_in_{k}"] = (df["days_to_next_holiday_closure"] == k).astype(float)
    for m in range(2, 13):  # January is the reference month
        X[f"month_{m}"] = (df["month"] == m).astype(float)
    X["trend_years"] = df["trend_years"].astype(float)
    if brand == "Style":
        X["style_peak_week"] = df["style_peak_week"].astype(float)
    return X


def recency_weights(dates: pd.Series, halflife_weeks):
    if halflife_weeks is None:
        return None
    age_weeks = (dates.max() - dates).dt.days / 7.0
    return (0.5 ** (age_weeks / halflife_weeks)).to_numpy()


def _window(train: pd.DataFrame, window_weeks):
    if window_weeks is None:
        return train
    return train[train["date"] > train["date"].max() - pd.Timedelta(weeks=window_weeks)]


class SeriesGLM:
    """Poisson GLM for one depot x brand series and one target column."""

    def __init__(self, brand: str, target: str = "total_volume", payday: str = "cal", alpha: float = GLM_ALPHA,
                 include_midweek: bool = True, halflife_weeks=None, window_weeks=None):
        self.brand, self.target, self.payday, self.alpha = brand, target, payday, alpha
        self.include_midweek, self.halflife_weeks, self.window_weeks = include_midweek, halflife_weeks, window_weeks

    def _X(self, df):
        return design_matrix(df, self.brand, self.payday, self.include_midweek)

    def fit(self, train: pd.DataFrame) -> "SeriesGLM":
        op = train[train["is_operating"] == 1]
        mean_by_dow = op.groupby("dow")[self.target].mean()
        self.zero_dows = sorted(int(d) for d in mean_by_dow.index
                                if mean_by_dow[d] < STRUCTURAL_ZERO_SHARE * op[self.target].mean())
        fit_rows = _window(op[~op["dow"].isin(self.zero_dows)], self.window_weeks)
        X = self._X(fit_rows)
        # Drop features that never vary in the training window (e.g. a weekday that is a structural zero)
        self.columns = [c for c in X.columns if X[c].std() > 0]
        self.model = PoissonRegressor(alpha=self.alpha, max_iter=5000, tol=1e-8)
        self.model.fit(X[self.columns], fit_rows[self.target],
                       sample_weight=recency_weights(fit_rows["date"], self.halflife_weeks))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._X(df)
        for c in self.columns:  # columns are fixed at fit time
            if c not in X:
                X[c] = 0.0
        pred = self.model.predict(X[self.columns])
        active = (df["is_operating"] == 1) & ~df["dow"].isin(self.zero_dows)
        return np.where(active, pred, 0.0)

    def multipliers(self) -> pd.Series:
        return pd.Series(np.exp(self.model.coef_), index=self.columns)


class StyleOutletGLM:
    """Poisson GLM on Style order volume per outlet-date (rows: dates of each outlet's
    weekday). Outlet dummies replace weekday dummies; calendar features as SeriesGLM."""

    def __init__(self, payday: str = "cal", alpha: float = GLM_ALPHA, include_midweek: bool = True,
                 halflife_weeks=None, window_weeks=None):
        self.payday, self.alpha, self.include_midweek = payday, alpha, include_midweek
        self.halflife_weeks, self.window_weeks = halflife_weeks, window_weeks

    def _X(self, df):
        X = design_matrix(df, "Style", self.payday, self.include_midweek, include_dow=False)
        for o in self.outlets[1:]:  # first outlet is the reference
            X[f"outlet_{o}"] = (df["outlet_id"] == o).astype(float)
        return X

    def fit(self, train: pd.DataFrame) -> "StyleOutletGLM":
        self.outlets = sorted(train["outlet_id"].unique())
        op = _window(train[train["is_operating"] == 1], self.window_weeks)
        X = self._X(op)
        self.columns = [c for c in X.columns if X[c].std() > 0]
        self.model = PoissonRegressor(alpha=self.alpha, max_iter=5000, tol=1e-8)
        self.model.fit(X[self.columns], op["order_volume_m3"], sample_weight=recency_weights(op["date"], self.halflife_weeks))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._X(df)
        for c in self.columns:
            if c not in X:
                X[c] = 0.0
        return np.where(df["is_operating"] == 1, self.model.predict(X[self.columns]), 0.0)


# ---------------------------------------------------------------- global HGB
HGB_NUMERIC = ["dow", "payday_cal_d0", "payday_cal_d1", "payday_cal_d2", "festival_ramp",
               "days_to_next_holiday_closure", "month", "trend_years", "style_peak_week"]
HGB_CATEGORICAL = {"depot": DEPOTS, "brand": BRANDS, "festival_name": ["none"] + FESTIVALS}


class GlobalHGB:
    """HistGradientBoostingRegressor (Poisson loss) across all depot x brand series."""

    def __init__(self, include_midweek: bool = True, **params):
        self.include_midweek = include_midweek
        self.params = dict(loss="poisson", max_depth=3, learning_rate=0.05, max_iter=400, min_samples_leaf=40,
                           l2_regularization=1.0, early_stopping=False, random_state=SEED)
        self.params.update(params)
        self.numeric = HGB_NUMERIC + (["after_midweek_closure"] if include_midweek else [])

    def _X(self, df):
        X = df[self.numeric].astype(float).copy()
        for c, levels in HGB_CATEGORICAL.items():
            codes = df[c].map({v: i for i, v in enumerate(levels)})
            assert codes.notna().all(), f"unknown {c} level"
            X[c] = codes.astype(int)
        return X

    def fit(self, train: pd.DataFrame, target: str = "total_volume") -> "GlobalHGB":
        op = train[train["is_operating"] == 1]
        X = self._X(op)
        self.model = HistGradientBoostingRegressor(categorical_features=[c in HGB_CATEGORICAL for c in X.columns],
                                                   **self.params)
        self.model.fit(X, op[target])
        # same structural zeros as the GLM, per series
        means = op.groupby(["depot", "brand", "dow"])[target].mean()
        series_mean = op.groupby(["depot", "brand"])[target].mean()
        self.zero = {k for k, v in means.items() if v < STRUCTURAL_ZERO_SHARE * series_mean[k[:2]]}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict(self._X(df))
        zero = np.array([(d, b, w) in self.zero for d, b, w in zip(df["depot"], df["brand"], df["dow"])])
        return np.where((df["is_operating"] == 1).to_numpy() & ~zero, pred, 0.0)


# ---------------------------------------------------------------- weekly smoothers (Tech)
def smooth_last(values: np.ndarray, kind: str, param: int) -> float:
    """Level from the most recent weekly values: kind "mean" (last `param` weeks) or
    "ewm" (exponential weights with half-life `param` weeks)."""
    if kind == "mean":
        return float(np.mean(values[-param:]))
    return float(pd.Series(values).ewm(halflife=param).mean().iloc[-1])


def weekday_weights(daily_train: pd.DataFrame, target: str = "total_volume") -> pd.Series:
    """Relative volume per operating weekday (mean 1 over operating days) for one series."""
    op = daily_train[daily_train["is_operating"] == 1]
    return op.groupby("dow")[target].mean() / op[target].mean()


def week_capacity(daily: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Sum of weekday weights over the operating days of each ISO week."""
    w = daily["dow"].map(weights).fillna(0.0) * daily["is_operating"]
    return daily.assign(cap=w).groupby(["iso_year", "iso_week"], as_index=False)["cap"].sum()


def calendar_level_forecast(daily_train, weekly_train, daily_future, kind, param, target="total_volume"):
    """Tech option (c): smoothed level per unit of weekday capacity, times each forecast
    week's capacity (so a week losing a busy weekday to a closure gets less volume)."""
    weights = weekday_weights(daily_train, target)
    cap_hist = week_capacity(daily_train, weights)
    wk = weekly_train.merge(cap_hist, on=["iso_year", "iso_week"])
    wk = wk[wk["cap"] > 0]
    level = smooth_last((wk[target] / wk["cap"]).to_numpy(), kind, param)
    cap_future = week_capacity(daily_future, weights)
    return cap_future.assign(pred=level * cap_future["cap"])


# ---------------------------------------------------------------- chilled
def chilled_share_table(daily_train: pd.DataFrame) -> pd.Series:
    """Fresh chilled share of total volume by depot x month, from training history."""
    f = daily_train[daily_train["brand"] == "Fresh"]
    g = f.groupby(["depot", "month"])[["chilled_volume", "total_volume"]].sum()
    return g["chilled_volume"] / g["total_volume"]


def weekly_sum(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    return df.groupby(WEEK_KEYS, as_index=False)[pred_col].sum()


def baseline_forecasts(weekly_hist: pd.DataFrame, target_weeks: pd.DataFrame, value: str) -> pd.DataFrame:
    """Weekly baselines for one series. weekly_hist: that series' training weeks
    (sorted); target_weeks: rows with iso_year, iso_week to forecast."""
    v = weekly_hist[value].to_numpy()
    out = target_weeks[["iso_year", "iso_week"]].copy()
    out["last8_mean"] = v[-8:].mean()
    out["mean13"] = v[-13:].mean()
    out["mean26"] = v[-26:].mean()
    out["ewm_hl8"] = pd.Series(v).ewm(halflife=8).mean().iloc[-1]
    ly = weekly_hist.set_index(["iso_year", "iso_week"])[value]
    out["same_week_last_year"] = [ly.get((y - 1, w), np.nan) for y, w in zip(out["iso_year"], out["iso_week"])]
    return out
