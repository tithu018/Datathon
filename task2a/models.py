"""Task 2A models: daily Poisson GLM per depot x brand series, plus weekly baselines.

The GLM is fitted on operating days only (log link, so coefficients are
multipliers). Closed days, and weekdays on which a series never takes orders
(e.g. Style on Tuesday), are forced to 0. Daily predictions are summed to ISO weeks.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

FESTIVALS = ["thai_pongal", "new_year", "vesak", "poson", "esala", "deepavali", "christmas"]
GLM_ALPHA = 1e-4          # light L2 penalty: keeps rare dummies finite, leaves multipliers ~unpenalized
STRUCTURAL_ZERO_SHARE = 0.01  # a weekday below 1% of the series' mean daily volume is "no delivery"


def design_matrix(df: pd.DataFrame, brand: str, payday: str = "cal") -> pd.DataFrame:
    """GLM features for one series. `payday` selects the payday tail counting:
    "cal" (calendar days) or "op" (operating days)."""
    X = pd.DataFrame(index=df.index)
    for d in range(1, 6):  # Monday is the reference weekday; Sunday is never operating
        X[f"dow_{d}"] = (df["dow"] == d).astype(float)
    for k in range(3):
        X[f"payday_d{k}"] = df[f"payday_{payday}_d{k}"].astype(float)
    for f in FESTIVALS:
        on = (df["festival_name"] == f).astype(float)
        X[f"ramp_{f}"] = df["festival_ramp"] * on
        X[f"ramp2_{f}"] = df["festival_ramp_sq"] * on
    X["after_midweek_closure"] = df["after_midweek_closure"].astype(float)
    for k in (1, 2, 3):  # 1-3 days before a non-Sunday closure (demand pulled forward)
        X[f"holiday_closure_in_{k}"] = (df["days_to_next_holiday_closure"] == k).astype(float)
    for m in range(2, 13):  # January is the reference month
        X[f"month_{m}"] = (df["month"] == m).astype(float)
    X["trend_years"] = df["trend_years"].astype(float)
    if brand == "Style":
        X["style_peak_week"] = df["style_peak_week"].astype(float)
    return X


class SeriesGLM:
    """Poisson GLM for one depot x brand series and one target column."""

    def __init__(self, brand: str, target: str = "total_volume", payday: str = "cal", alpha: float = GLM_ALPHA):
        self.brand, self.target, self.payday, self.alpha = brand, target, payday, alpha

    def fit(self, train: pd.DataFrame) -> "SeriesGLM":
        op = train[train["is_operating"] == 1]
        mean_by_dow = op.groupby("dow")[self.target].mean()
        self.zero_dows = sorted(int(d) for d in mean_by_dow.index
                                if mean_by_dow[d] < STRUCTURAL_ZERO_SHARE * op[self.target].mean())
        fit_rows = op[~op["dow"].isin(self.zero_dows)]
        X = design_matrix(fit_rows, self.brand, self.payday)
        # Drop features that never vary in the training window (e.g. a weekday that is a structural zero)
        self.columns = [c for c in X.columns if X[c].std() > 0]
        self.model = PoissonRegressor(alpha=self.alpha, max_iter=5000, tol=1e-8)
        self.model.fit(X[self.columns], fit_rows[self.target])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = design_matrix(df, self.brand, self.payday)
        for c in self.columns:  # columns are fixed at fit time
            if c not in X:
                X[c] = 0.0
        pred = self.model.predict(X[self.columns])
        active = (df["is_operating"] == 1) & ~df["dow"].isin(self.zero_dows)
        return np.where(active, pred, 0.0)

    def multipliers(self) -> pd.Series:
        return pd.Series(np.exp(self.model.coef_), index=self.columns)


def weekly_sum(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    return df.groupby(["depot", "brand", "iso_year", "iso_week"], as_index=False)[pred_col].sum()


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
