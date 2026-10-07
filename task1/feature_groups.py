"""Task 1 Phase 7 feature groups. Every feature is known at planning time: built from
outlets.csv, service_allowance.csv, calendar.csv, traffic_speed.csv, road_conditions.csv
(dispatch-day district disruption; known on the dispatch morning) and the PLANNED
columns of the route legs (planned_depart_time, planned_travel_duration_min,
planned_arrival_time). Actual times are never used.
"""
from pathlib import Path

import numpy as np
import pandas as pd

GENERAL = Path("General Data")


def _minutes(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, format="%H:%M", errors="coerce")
    assert t.notna().all(), "unparseable HH:MM time"
    return t.dt.hour * 60 + t.dt.minute


def route_context(legs: pd.DataFrame, outlets: pd.DataFrame) -> pd.DataFrame:
    """Per route leg (route_id, seq): planned dwell, route size, cumulative planned travel,
    planned route start, and the waits the plan did not schedule at earlier stops."""
    L = legs[["route_id", "seq", "brand", "to_outlet", "planned_depart_time", "planned_travel_duration_min",
              "planned_arrival_time"]].copy()
    L = L.sort_values(["route_id", "seq"])
    L["pa"] = _minutes(L["planned_arrival_time"])
    L["pd"] = _minutes(L["planned_depart_time"])
    g = L.groupby("route_id")
    # Planned dwell: the next leg's planned departure from this outlet minus the planned arrival here
    L["planned_dwell_min"] = g["pd"].shift(-1) - L["pa"]
    L["is_last_stop"] = L["planned_dwell_min"].isna().astype(int)
    L["n_stops"] = g["seq"].transform("size")
    L["cum_planned_travel_min"] = g["planned_travel_duration_min"].cumsum()  # up to and including this leg
    L["planned_route_start_min"] = g["pd"].transform("first")
    L = L.merge(outlets[["outlet_id", "window_open_time"]], left_on="to_outlet", right_on="outlet_id", how="left")
    L = L.sort_values(["route_id", "seq"])
    # Phase 7 finding: the planned dwell equals the service allowance at every stop that has a next
    # leg, so "cumulative allowance of earlier stops minus planned dwell" was identically 0 and was
    # dropped. The plan does NOT schedule the wait when a vehicle arrives before the window opens;
    # the sum of those waits at earlier stops is measured as a labelled extra only (not applied).
    wait = (_minutes(L["window_open_time"]) - L["pa"]).clip(lower=0)
    L["cum_unplanned_wait_before"] = wait.groupby(L["route_id"]).cumsum() - wait
    L["planned_dwell_min"] = L["planned_dwell_min"].fillna(-1)  # last stop: no next leg
    return L[["route_id", "seq", "planned_dwell_min", "is_last_stop", "n_stops", "cum_planned_travel_min",
              "planned_route_start_min", "cum_unplanned_wait_before"]]


def add_feature_groups(df: pd.DataFrame, legs: pd.DataFrame) -> pd.DataFrame:
    """df: one row per order with outlet_id, brand, district, route_id, seq, the leg date
    ('date') and the planned-minute columns from prepare_features.py."""
    outlets = pd.read_csv(GENERAL / "outlets.csv")
    allowance = pd.read_csv(GENERAL / "service_allowance.csv")
    calendar = pd.read_csv(GENERAL / "calendar.csv")
    traffic = pd.read_csv(GENERAL / "traffic_speed.csv")
    roads = pd.read_csv(GENERAL / "road_conditions.csv")
    n = len(df)

    # A. outlet attributes
    o = outlets[["outlet_id", "dock_type", "parking_constraint", "mall_window"]].copy()
    mw = o["mall_window"].str.split("-", expand=True)
    o["mall_window_open_min"] = np.where(o["mall_window"].notna(), _minutes(mw[0].fillna("00:00")), -1)
    o["mall_window_close_min"] = np.where(o["mall_window"].notna(), _minutes(mw[1].fillna("00:00")), -1)
    df = df.merge(o.drop(columns="mall_window"), on="outlet_id", how="left")

    # B. dispatcher's service allowance (brand x dock type)
    df = df.merge(allowance, on=["brand", "dock_type"], how="left")

    # C/D. planned slack and route context (planned columns only)
    df = df.merge(route_context(legs, outlets), on=["route_id", "seq"], how="left")
    df["slack_to_close_min"] = df["window_close_min_feature"] - df["planned_arrival_min"]
    df["slack_from_open_min"] = df["planned_arrival_min"] - df["window_open_min_feature"]

    # E. calendar on the day the route ran
    cal = calendar[["date", "is_payday", "festival_ramp", "is_holiday"]]
    df = df.merge(cal, on="date", how="left")

    # F. typical traffic: district x planned arrival hour x monsoon
    df["planned_arrival_hour"] = (df["planned_arrival_min"] // 60).astype(int)
    df = df.merge(traffic.rename(columns={"hour": "planned_arrival_hour"}),
                  on=["district", "planned_arrival_hour", "monsoon"], how="left")

    # G. road conditions for the district on the dispatch day (user decision, Phase 7: predictions
    #    are made on the dispatch morning, when road advisories are known)
    df = df.merge(roads, on=["district", "date"], how="left")

    assert len(df) == n, "feature joins changed the row count"
    return df
