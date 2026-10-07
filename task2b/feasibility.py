"""Published rule checks, exhaustive chilled trips and submission helpers."""
from dataclasses import dataclass
from itertools import combinations
import subprocess
import sys

import pandas as pd

from task2b.analysis import compatible
from task2b.common import (FRESH_BUDGET_MIN, DAYTIME_BUDGET_MIN, MAX_TRIPS,
                           SUBMISSION_COLS, TEMPLATE, CHECKER, load_travel,
                           load_allowance, trip_time)
from task2b.policy import protected, volume_units


@dataclass(frozen=True)
class Trip:
    indices: tuple
    mask: int
    minutes: int
    urgent: int
    volume: int

    @property
    def score(self):
        return (self.urgent, self.volume, len(self.indices), -self.minutes)


def feasible_trip(group, vehicle, travel, allowance):
    if group.empty:
        return False
    if group.brand.nunique() != 1 or group.district.nunique() != 1:
        return False
    if not all(compatible(r, vehicle) for r in group.itertuples()):
        return False
    budget = FRESH_BUDGET_MIN if group.brand.iloc[0] == "Fresh" else DAYTIME_BUDGET_MIN
    return (group.order_volume_m3.sum() <= vehicle.volume_cap_m3 + 1e-6
            and group.order_weight_kg.sum() <= vehicle.weight_cap_kg + 1e-6
            and trip_time(group.district.iloc[0], group.brand.iloc[0], list(group.dock_type), travel, allowance) <= budget)


def enumerate_trips(orders, vehicle, travel, allowance):
    """Every nonempty compatible chilled subset, district/brand homogeneous."""
    trips = []
    chilled = orders[orders.temp_requirement.eq("chilled")]
    for (_, district), group in chilled.groupby(["brand", "district"]):
        indices = [r.Index for r in group.itertuples() if compatible(r, vehicle)]
        for n in range(1, len(indices) + 1):
            for subset in combinations(indices, n):
                g = orders.loc[list(subset)]
                if feasible_trip(g, vehicle, travel, allowance):
                    rows = list(g.itertuples())
                    trips.append(Trip(subset, sum(1 << i for i in subset),
                                      int(trip_time(district, g.brand.iloc[0], list(g.dock_type), travel, allowance)),
                                      sum(protected(r) for r in rows), sum(volume_units(r) for r in rows)))
    return sorted(trips, key=lambda t: (t.score, t.indices), reverse=True)


def make_submission(orders, allocation):
    sub = pd.read_csv(TEMPLATE)
    assert sub.columns.tolist() == SUBMISSION_COLS
    assert sub[["scenario", "order_ref", "outlet_id"]].equals(orders[["scenario", "order_ref", "outlet_id"]])
    sub["decision"], sub["vehicle_id"], sub["trip_id"] = "deferred", "", ""
    assigned = set()
    for vehicle_id, trips in sorted(allocation.items()):
        for tid, indices in enumerate(trips, 1):
            assert not assigned.intersection(indices), "duplicate order assignment"
            assigned.update(indices)
            sub.loc[list(indices), ["decision", "vehicle_id", "trip_id"]] = ["served", vehicle_id, str(tid)]
    return sub


def validate(sub, orders, fleet):
    """Strict submission checks, including identifiers the official checker omits."""
    assert sub.columns.tolist() == SUBMISSION_COLS
    assert sub[["scenario", "order_ref", "outlet_id"]].equals(orders[["scenario", "order_ref", "outlet_id"]])
    assert sub.decision.isin(["served", "deferred"]).all()
    deferred = sub[sub.decision.eq("deferred")]
    assert deferred[["vehicle_id", "trip_id"]].fillna("").eq("").all().all()
    served = sub[sub.decision.eq("served")]
    assert served[["vehicle_id", "trip_id"]].notna().all().all()
    assert pd.to_numeric(served.trip_id, errors="coerce").isin([1, 2]).all()
    travel, allowance = load_travel(), load_allowance()
    vehicles = {v.vehicle_id: v for v in fleet.itertuples()}
    total = 0
    for vid, vg in served.groupby("vehicle_id"):
        assert vid in vehicles and vehicles[vid].available
        fresh = daytime = 0
        assert vg.trip_id.nunique() <= MAX_TRIPS
        for _, g in vg.groupby("trip_id"):
            group = orders.loc[g.index]
            assert feasible_trip(group, vehicles[vid], travel, allowance), (vid, g.order_ref.tolist())
            minutes = trip_time(group.district.iloc[0], group.brand.iloc[0], list(group.dock_type), travel, allowance)
            total += minutes
            if group.brand.iloc[0] == "Fresh":
                fresh += minutes
            else:
                daytime += minutes
        assert fresh <= FRESH_BUDGET_MIN and daytime <= DAYTIME_BUDGET_MIN
    return int(total)


def official_check(path):
    result = subprocess.run([sys.executable, str(CHECKER), str(path)], capture_output=True, text=True)
    assert result.returncode == 0 and "PASSED" in result.stdout, result.stdout + result.stderr
    return result.stdout.strip()
