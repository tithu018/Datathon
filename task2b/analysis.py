"""Phase 1: capacity bounds and indivisible-order feasibility, no allocation."""
import sys
from itertools import combinations

sys.path.insert(0, ".")
from task2b.common import (REPORT_DIR, FRESH_BUDGET_MIN, DAYTIME_BUDGET_MIN,
                           load_orders, load_fleet, load_travel, load_allowance, trip_time)


def compatible(order, vehicle):
    return (order.scenario == vehicle.scenario and order.depot == vehicle.depot
            and (order.temp_requirement != "chilled" or vehicle.is_reefer)
            and (order.parking_constraint != "van_only" or vehicle.is_van))


def individually_feasible(order, vehicle, travel, allowance):
    budget = FRESH_BUDGET_MIN if order.brand == "Fresh" else DAYTIME_BUDGET_MIN
    return (compatible(order, vehicle)
            and order.order_volume_m3 <= vehicle.volume_cap_m3 + 1e-6
            and order.order_weight_kg <= vehicle.weight_cap_kg + 1e-6
            and trip_time(order.district, order.brand, [order.dock_type], travel, allowance) <= budget)


def chilled_bound(orders, fleet, travel, allowance):
    """Safe upper bound: best single/pair per vehicle, relaxing inter-vehicle overlap."""
    bounds = {}
    chilled = orders[orders.temp_requirement.eq("chilled")]
    for v in fleet[fleet.available & fleet.is_reefer].itertuples():
        trips = [(0, 0.0)]
        for (_, district), group in chilled.groupby(["brand", "district"]):
            rows = [r for r in group.itertuples() if compatible(r, v)]
            for n in range(1, len(rows) + 1):
                for subset in combinations(rows, n):
                    volume = sum(r.order_volume_m3 for r in subset)
                    weight = sum(r.order_weight_kg for r in subset)
                    minutes = trip_time(district, "Fresh", [r.dock_type for r in subset], travel, allowance)
                    if volume <= v.volume_cap_m3 + 1e-6 and weight <= v.weight_cap_kg + 1e-6 and minutes <= FRESH_BUDGET_MIN:
                        trips.append((minutes, volume))
        # Duplicate orders across the two trips are allowed ONLY in this relaxation.
        bounds[v.vehicle_id] = max(a[1] + b[1] for a in trips for b in trips
                                   if a[0] + b[0] <= FRESH_BUDGET_MIN)
    return bounds


def main():
    orders, fleet = load_orders(), load_fleet()
    travel, allowance = load_travel(), load_allowance()
    available = fleet[fleet.available]
    lines = ["Task 2B Phase 1: capacity analysis (S1)",
             "Published rules: whole orders; max 2 trips; 270 Fresh / 480 daytime minutes.",
             "Each order counts as one stop; no return time in the published duration.", "",
             "Demand by brand and temperature (orders, m3, kg):"]
    lines.append(orders.groupby(["brand", "temp_requirement"]).agg(
        orders=("order_ref", "size"), m3=("order_volume_m3", "sum"), kg=("order_weight_kg", "sum")
    ).to_string(float_format=lambda x: f"{x:.3f}"))
    lines += ["", "Available fleet by class (vehicles, one-trip m3, one-trip kg, max trips):"]
    classes = available.groupby(["temp", "type"]).agg(
        vehicles=("vehicle_id", "size"), m3=("volume_cap_m3", "sum"), kg=("weight_cap_kg", "sum"))
    classes["max_trips"] = 2 * classes.vehicles
    lines.append(classes.to_string())
    lines += ["", "District stop limits, using the smallest allowance for each brand.",
              "These ignore capacity, window waiting and other trips; they are upper bounds.",
              "district | outbound min | Fresh single min / max stops | Style single / max | Tech single / max"]
    for district in sorted(orders.district.unique()):
        d = travel[district]
        parts = []
        for brand in ("Fresh", "Style", "Tech"):
            service = min(v for (b, _), v in allowance.items() if b == brand)
            budget = FRESH_BUDGET_MIN if brand == "Fresh" else DAYTIME_BUDGET_MIN
            single = d["depot_to_district_freeflow_min"] + service
            max_stops = max(0, int((budget - d["depot_to_district_freeflow_min"]
                                  + d["inter_stop_freeflow_min"]) // (service + d["inter_stop_freeflow_min"])))
            parts.append(f"{single:.0f} / {max_stops}")
        lines.append(f"{district} | {d['depot_to_district_freeflow_min']} | " + " | ".join(parts))
    chilled = orders[orders.temp_requirement.eq("chilled")]
    reefers = available[available.is_reefer]
    volume_cap = 2 * reefers.volume_cap_m3.sum()
    bounds = chilled_bound(orders, fleet, travel, allowance)
    upper = min(volume_cap, sum(bounds.values()), chilled.order_volume_m3.sum())
    lines += ["", "BOTTLENECK 1: reefer volume, trip count, weight and Fresh time.",
              chilled.groupby(["district", "parking_constraint"])[["order_volume_m3", "order_weight_kg"]].sum().to_string(),
              f"Chilled demand: {chilled.order_volume_m3.sum():.3f} m3 / {chilled.order_weight_kg.sum():.1f} kg.",
              f"All four reefers, including van: at most {volume_cap:.3f} m3 in 8 trips.",
              f"Per-vehicle time/packing relaxed volume bounds: {bounds}",
              f"Safe served chilled upper bound: {upper:.3f} m3; unavoidable aggregate shortfall >= {chilled.order_volume_m3.sum()-upper:.3f} m3.",
              "The bounds may reuse the same order; they are not a feasible allocation or an optimality certificate.",
              "Three trucks: 158.400 m3 / 31,920 kg across at most six trips; seven normal-access chilled districts.",
              "The van can serve normal-access orders too, so a six-truck-trip district argument alone is not a universal proof.",
              "Van-only chilled demand 6.013 m3 / 1,095.7 kg exceeds the van's 1,040 kg single-trip limit; all three require two trips.",
              "", "BOTTLENECK 2: indivisible orders that fit no available vehicle."]
    impossible = []
    for order in orders.itertuples():
        if not any(individually_feasible(order, v, travel, allowance) for v in available.itertuples()):
            impossible.append(order.order_ref)
            lines.append(f"{order.order_ref} ({order.brand}, {order.district}): {order.order_volume_m3:.3f} m3 / {order.order_weight_kg:.1f} kg; no compatible available vehicle.")
    assert impossible == ["S1-078"], impossible
    lines += ["Largest available ambient truck: 38.000 m3. S1-078 is 40.660 m3; it cannot be split.",
              "Aggregate ambient capacity is ample, but this specific Style order must be deferred.",
              "All other orders fit some compatible vehicle individually; joint feasibility remains to be solved."]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = "\n".join(line.rstrip() for line in "\n".join(lines).splitlines()) + "\n"
    (REPORT_DIR / "capacity_analysis.txt").write_text(output, encoding="utf-8")
    print(output)
    print("Phase 1 checks PASSED")


if __name__ == "__main__":
    main()
