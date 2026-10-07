"""Phase 5: strict identifiers, fairness and approximate operational diagnostics.

Window simulation retains the published no-return convention. It does not change
the allocator's feasible set or claim a physically executable vehicle schedule.
"""
import sys
from itertools import permutations
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from task2b.common import (DATA_DIR, REPORT_DIR, load_orders, load_fleet,
                           load_travel, load_allowance, trip_time)
from task2b.feasibility import validate, official_check
from task2b.policy import protected


def clock_minutes(value):
    hour, minute = str(value).split(":")
    return int(hour) * 60 + int(minute)


def clock_text(value):
    value = int(round(value))
    return f"{value // 60:02d}:{value % 60:02d}"


def effective_window(row):
    opening = clock_minutes(row.window_open_time)
    closing = clock_minutes(row.window_close_time)
    if row.parking_constraint == "mall_dock" and pd.notna(row.mall_window):
        mall_open, mall_close = row.mall_window.split("-")
        opening = max(opening, clock_minutes(mall_open))
        closing = min(closing, clock_minutes(mall_close))
    assert opening <= closing, row.order_ref
    return opening, closing


def sequence_trip(group, departure, travel, allowance, require_on_time):
    """Exact earliest-finish subset DP; district inter-stop times are identical.

    Earliest completion dominates a later completion for the same visited subset:
    the next leg takes the same time regardless of the last outlet. Thus one state
    per subset suffices, including when enforcing every raw arrival <= close.
    """
    rows = list(group.sort_values("order_ref").itertuples())
    assert rows and len({r.district for r in rows}) == 1
    district = travel[rows[0].district]
    states = {0: (float(departure), ())}
    full = (1 << len(rows)) - 1
    for mask in range(full + 1):
        if mask not in states:
            continue
        finish, sequence = states[mask]
        leg = (district["depot_to_district_freeflow_min"] if mask == 0
               else district["inter_stop_freeflow_min"])
        arrival = finish + leg
        for i, row in enumerate(rows):
            if mask & (1 << i):
                continue
            opening, closing = effective_window(row)
            if require_on_time and arrival > closing + 1e-6:
                continue
            end = max(arrival, opening) + allowance[(row.brand, row.dock_type)]
            candidate = (end, sequence + (i,))
            next_mask = mask | (1 << i)
            if next_mask not in states or candidate < states[next_mask]:
                states[next_mask] = candidate
    if full not in states:
        return None
    finish, sequence = states[full]
    current = float(departure)
    timeline = []
    for position, i in enumerate(sequence, 1):
        row = rows[i]
        opening, closing = effective_window(row)
        leg = (district["depot_to_district_freeflow_min"] if position == 1
               else district["inter_stop_freeflow_min"])
        arrival = current + leg
        start = max(arrival, opening)
        end = start + allowance[(row.brand, row.dock_type)]
        timeline.append({"order_ref": row.order_ref, "outlet_id": row.outlet_id,
                         "brand": row.brand, "district": row.district,
                         "sequence": position, "departure_min": current,
                         "travel_min": leg, "arrival_min": arrival,
                         "window_open_min": opening, "window_close_min": closing,
                         "early_arrival": arrival < opening,
                         "wait_min": start - arrival, "service_start_min": start,
                         "service_end_min": end, "arrival_late": arrival > closing,
                         "finish_after_close": end > closing})
        current = end
    assert abs(current - finish) < 1e-6
    return finish, timeline


def sequence_window(trips, departure, travel, allowance):
    """Try both trip orders, with arrival feasibility before fallback diagnostics."""
    if not trips:
        return [], True
    for require_on_time in (True, False):
        candidates = []
        for trip_order in permutations(trips):
            current, timeline = departure, []
            for trip_id, group in trip_order:
                result = sequence_trip(group, current, travel, allowance, require_on_time)
                if result is None:
                    break
                current, records = result
                for record in records:
                    record["trip_id"] = int(trip_id)
                timeline.extend(records)
            else:
                candidates.append((current, tuple(r["order_ref"] for r in timeline), timeline))
        if candidates:
            return min(candidates, key=lambda item: item[:2])[2], require_on_time
    raise AssertionError("unconstrained window DP must have a solution")


def audit_plan(path, label, orders, fleet, travel, allowance):
    sub = pd.read_csv(path)
    published_minutes = validate(sub, orders, fleet)
    official = official_check(path)
    merged = orders.copy()
    for column in ("decision", "vehicle_id", "trip_id"):
        merged[column] = sub[column]
    served = merged[merged.decision.eq("served")]
    deferred = merged[merged.decision.eq("deferred")]
    vehicles = {v.vehicle_id: v for v in fleet.itertuples()}
    records, window_failures = [], 0
    for vehicle_id, group in served.groupby("vehicle_id"):
        trips = list(group.groupby("trip_id"))
        fresh = [(tid, g) for tid, g in trips if g.brand.iloc[0] == "Fresh"]
        daytime = [(tid, g) for tid, g in trips if g.brand.iloc[0] != "Fresh"]
        morning, ok = sequence_window(fresh, 210, travel, allowance)
        window_failures += int(not ok)
        morning_end = max((r["service_end_min"] for r in morning), default=210)
        # One vehicle cannot start its daytime trip while still completing Fresh.
        day, ok = sequence_window(daytime, max(540, morning_end), travel, allowance)
        window_failures += int(not ok)
        for record in morning + day:
            record["vehicle_id"] = vehicle_id
        records.extend(morning + day)
    timeline = pd.DataFrame(records)
    assert len(timeline) == len(served)
    assert set(timeline.order_ref) == set(served.order_ref)
    for field in ("departure", "arrival", "service_start", "service_end"):
        timeline[field + "_time"] = timeline[field + "_min"].map(clock_text)
    timeline.to_csv(DATA_DIR / f"{label}_timeline.csv", index=False)

    manifest = []
    for (vid, tid), group in served.groupby(["vehicle_id", "trip_id"]):
        v = vehicles[vid]
        district = travel[group.district.iloc[0]]
        route = timeline[(timeline.vehicle_id == vid) & timeline.trip_id.eq(int(tid))]
        outbound = district["depot_to_district_km"]
        between = (len(group) - 1) * district["inter_stop_km"]
        estimate_km = outbound + between + outbound
        manifest.append({"vehicle_id": vid, "trip_id": int(tid),
                         "brand": group.brand.iloc[0], "district": group.district.iloc[0],
                         "orders": len(group), "volume_m3": group.order_volume_m3.sum(),
                         "weight_kg": group.order_weight_kg.sum(),
                         "published_minutes": trip_time(group.district.iloc[0], group.brand.iloc[0],
                                                       list(group.dock_type), travel, allowance),
                         "departure_min": route.iloc[0].departure_min,
                         "finish_min": route.iloc[-1].service_end_min,
                         "wait_min": route.wait_min.sum(), "outbound_km": outbound,
                         "inter_stop_km": between, "estimated_return_km": outbound,
                         "estimated_total_km": estimate_km,
                         "estimated_fuel_l": estimate_km / v.km_per_l})
    trips = pd.DataFrame(manifest)
    trips.to_csv(DATA_DIR / f"{label}_trip_manifest.csv", index=False)
    fuel = trips.groupby("vehicle_id").agg(estimated_total_km=("estimated_total_km", "sum"),
                                           estimated_fuel_l=("estimated_fuel_l", "sum"))
    fuel["weekly_quota_l"] = [vehicles[vid].weekly_fuel_quota_l for vid in fuel.index]
    fuel["one_day_percent_of_weekly_quota"] = 100 * fuel.estimated_fuel_l / fuel.weekly_quota_l
    fuel["one_day_exceeds_weekly_quota"] = fuel.estimated_fuel_l > fuel.weekly_quota_l
    fuel.to_csv(DATA_DIR / f"{label}_fuel.csv")

    urgent = merged.apply(lambda row: protected(row), axis=1)
    assert int(urgent.sum()) == 10
    covered_outlets = set(served.outlet_id)
    protected_outlets = set(merged.loc[urgent, "outlet_id"])
    chilled_deferred = deferred[deferred.temp_requirement.eq("chilled")]
    ambient_at_pairs = merged[merged.temp_requirement.eq("ambient")
                             & merged.outlet_id.isin(chilled_deferred.outlet_id)]
    assert ambient_at_pairs.decision.eq("served").all()
    all_fresh = timeline[timeline.brand.eq("Fresh")]
    lines = [f"{label}: {official}",
             f"Orders served/deferred: {len(served)} / {len(deferred)}; outlets covered: {len(covered_outlets)} / {merged.outlet_id.nunique()}.",
             f"Protected order rows served: {int((urgent & merged.decision.eq('served')).sum())} / {int(urgent.sum())}.",
             f"Protected outlets with at least one delivery: {len(protected_outlets & covered_outlets)} / {len(protected_outlets)}.",
             f"Deferred chilled orders: {len(chilled_deferred)}; paired ambient orders served: {len(ambient_at_pairs)} / {len(ambient_at_pairs)}.",
             f"Chilled served: {served.loc[served.temp_requirement.eq('chilled'), 'order_volume_m3'].sum():.3f} m3.",
             f"Published vehicle-minutes: {published_minutes}; simulated window waiting: {timeline.wait_min.sum():.0f} minutes.",
             f"Vehicle/window groups with no all-arrivals-on-time ordering: {window_failures}.",
             f"Raw early arrivals: {int(timeline.early_arrival.sum())}; arrival after close: {int(timeline.arrival_late.sum())}; service finish after close: {int(timeline.finish_after_close.sum())}.",
             f"Fresh service completions after 08:00: {int(all_fresh.service_end_min.gt(480).sum())}.",
             f"Estimated distance including approximate returns: {fuel.estimated_total_km.sum():.1f} km; estimated fuel: {fuel.estimated_fuel_l.sum():.1f} L.",
             f"Vehicles whose one-day estimate alone exceeds weekly quota: {int(fuel.one_day_exceeds_weekly_quota.sum())}."]
    return lines


def main():
    orders, fleet = load_orders(), load_fleet()
    travel, allowance = load_travel(), load_allowance()
    reference_files = [Path("General Data") / name for name in
                       ("vehicles.csv", "district_travel.csv", "service_allowance.csv")]
    reference_files += [Path("Test Data") / name for name in
                        ("task2b_peak_day_scenarios.csv", "task2b_peak_day_fleet.csv")]
    for path in reference_files:
        assert path.read_bytes() == (Path("data") / path).read_bytes(), f"checker data differs: {path}"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["Task 2B Phase 5: independent allocation audit", "",
             "All five allocator/checker input files are identical byte-for-byte.",
             "Strict identifiers, decisions, blank deferred assignments and published feasibility are asserted.",
             "Operational approximation: Fresh starts at 03:30; daytime starts at 09:00 or after Fresh completion.",
             "Exact subset DP searches stop order and both trip orders for on-time raw arrivals, then earliest finish.",
             "Wait until windows open; mall windows intersect outlet windows. Arrival after close defines lateness.",
             "Service finishing after close is reported separately. Windows are diagnostics, not optimizer constraints.",
             "No return travel or reloading duration in this timeline, following the published planning convention.",
             "This is not a physical execution guarantee: the booklet budgets already allow for returns.",
             "Fuel uses outbound + inter-stop distance + an estimated return equal to outbound distance.",
             "One-day fuel versus a weekly quota cannot establish weekly compliance without prior consumption.", ""]
    for name in ("optimized", "volume_first"):
        lines += audit_plan(DATA_DIR / f"{name}_submission.csv", name, orders, fleet, travel, allowance)
        lines.append("")
    lines += ["Order histories can differ between chilled and ambient rows at one outlet.",
              "Protected order fulfillment and outlet coverage are therefore separate measures.",
              "Detailed timelines, trip manifests and fuel estimates are local under ignored task2b/data/.",
              "Audit checks PASSED"]
    output = "\n".join(lines) + "\n"
    (REPORT_DIR / "audit_report.txt").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
