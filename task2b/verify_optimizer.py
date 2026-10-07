"""Independent tiny-instance exhaustive oracle for the exact chilled search.

Uses only stdlib/pandas and the supplied input tables. The oracle assigns each
order directly to a vehicle and trip; it does not call production enumeration,
schedule dominance, feasibility helpers or branch bounds.
"""
import random
import sys

sys.path.insert(0, ".")
from task2b.common import load_orders, load_fleet, load_travel, load_allowance
from task2b.optimize import solve


def brute_force(orders, fleet):
    vehicles = list(fleet[fleet.available & fleet.is_reefer].itertuples())
    rows = list(orders[orders.temp_requirement.eq("chilled")].itertuples())
    travel, allowance = load_travel(), load_allowance()
    trips = [[None, None] for _ in vehicles]
    best_priority = (-1, -1, -1, -10**9)
    best_volume = (-1, -1, -1, -10**9)
    nodes = 0

    def duration(trip):
        if trip is None:
            return 0
        district, _, _, _, handling, stops = trip
        d = travel[district]
        return (d["depot_to_district_freeflow_min"]
                + (stops - 1) * d["inter_stop_freeflow_min"] + handling)

    def visit(index, urgent, volume, count):
        nonlocal best_priority, best_volume, nodes
        nodes += 1
        if index == len(rows):
            minutes = sum(duration(t) for pair in trips for t in pair)
            best_priority = max(best_priority, (urgent, volume, count, -minutes))
            best_volume = max(best_volume, (volume, urgent, count, -minutes))
            return
        r = rows[index]
        visit(index + 1, urgent, volume, count)  # defer the whole order
        for vi, vehicle in enumerate(vehicles):
            if (r.scenario != vehicle.scenario or r.depot != vehicle.depot
                    or (r.parking_constraint == "van_only" and not vehicle.is_van)):
                continue
            for ti in range(2):
                # Trips are interchangeable: the earliest assigned order starts
                # trip 1. This removes permutations without discarding a plan.
                if ti == 1 and trips[vi][0] is None:
                    continue
                old = trips[vi][ti]
                if old is not None and old[:2] != (r.district, r.brand):
                    continue
                old_volume, old_weight, old_handling, old_count = (
                    old[2:] if old is not None else (0, 0, 0, 0))
                new = (r.district, r.brand, old_volume + r.order_volume_m3,
                       old_weight + r.order_weight_kg,
                       old_handling + allowance[(r.brand, r.dock_type)], old_count + 1)
                if (new[2] > vehicle.volume_cap_m3 + 1e-6
                        or new[3] > vehicle.weight_cap_kg + 1e-6
                        or duration(new) + duration(trips[vi][1-ti]) > 270):
                    continue
                trips[vi][ti] = new
                visit(index + 1,
                      urgent + int(r.deferred_yesterday == 1 or r.days_since_last_served >= 3),
                      volume + round(r.order_volume_m3 * 1000), count + 1)
                trips[vi][ti] = old

    visit(0, 0, 0, 0)
    return best_priority, best_volume, nodes


def tiny_instance(seed):
    rng = random.Random(seed)
    original = load_orders()
    size = 5 + seed % 2
    orders = original.iloc[:size].copy()
    # Nonconsecutive positive indices exercise masks using original row labels.
    orders.index = [1 + 3*i for i in range(size)]
    for position, i in enumerate(orders.index):
        orders.loc[i, ["scenario", "depot", "brand", "temp_requirement"]] = [
            "S2" if rng.random() < .12 else "S1",
            "Other" if rng.random() < .12 else "Peliyagoda", "Fresh", "chilled"]
        orders.loc[i, "district"] = rng.choice(["Colombo", "Gampaha", "Matara", "Puttalam"])
        orders.loc[i, "dock_type"] = rng.choice(["street", "rear_dock"])
        orders.loc[i, "parking_constraint"] = rng.choice(["normal", "normal", "van_only"])
        orders.loc[i, "order_volume_m3"] = rng.choice([1.001, 2.245, 3.0, 4.5, 7.0, 10.5])
        orders.loc[i, "order_weight_kg"] = rng.choice([300.0, 800.0, 1040.0, 1100.0, 1900.0])
        # Include independent flags and orders with both; urgency must count once.
        orders.loc[i, "deferred_yesterday"] = position % 2
        orders.loc[i, "days_since_last_served"] = 1 if position % 3 == 0 else 3
        orders.loc[i, "order_ref"] = f"T{seed}-{position}"
        orders.loc[i, "outlet_id"] = f"O{position}"
    fleet = load_fleet().query("available and is_reefer").copy()
    if seed % 2 == 0:
        fleet = fleet[fleet.vehicle_id.ne("VEH007")].copy()
    for i, vehicle in fleet.iterrows():
        if vehicle.is_van:
            fleet.loc[i, ["volume_cap_m3", "weight_cap_kg"]] = [7.0, 1040]
        else:
            fleet.loc[i, ["volume_cap_m3", "weight_cap_kg"]] = [
                rng.choice([4.5, 7.0, 12.0]), rng.choice([1040, 1900, 3000])]
    return orders, fleet


def witness_key(orders, fleet, allocation, volume_first):
    """Recompute the returned assignment rather than trust its certificate."""
    vehicles = {v.vehicle_id: v for v in fleet.itertuples()}
    travel, allowance = load_travel(), load_allowance()
    assigned = set()
    total_minutes = 0
    for vid, trips in allocation.items():
        v = vehicles[vid]
        assert v.available and v.is_reefer and len(trips) <= 2
        vehicle_minutes = 0
        for indices in trips:
            assert indices and not assigned.intersection(indices)
            assigned.update(indices)
            rows = list(orders.loc[list(indices)].itertuples())
            assert len({(r.brand, r.district) for r in rows}) == 1
            assert all(r.temp_requirement == "chilled" and r.scenario == v.scenario
                       and r.depot == v.depot
                       and (r.parking_constraint != "van_only" or v.is_van) for r in rows)
            assert sum(r.order_volume_m3 for r in rows) <= v.volume_cap_m3 + 1e-6
            assert sum(r.order_weight_kg for r in rows) <= v.weight_cap_kg + 1e-6
            d = travel[rows[0].district]
            vehicle_minutes += (d["depot_to_district_freeflow_min"]
                                + (len(rows)-1)*d["inter_stop_freeflow_min"]
                                + sum(allowance[(r.brand, r.dock_type)] for r in rows))
        assert vehicle_minutes <= 270
        total_minutes += vehicle_minutes
    rows = list(orders.loc[sorted(assigned)].itertuples())
    urgency = sum(r.deferred_yesterday == 1 or r.days_since_last_served >= 3 for r in rows)
    volume = sum(round(r.order_volume_m3*1000) for r in rows)
    leading = (volume, urgency) if volume_first else (urgency, volume)
    return (*leading, len(rows), -total_minutes)


def main():
    nodes = 0
    instances = []
    for seed in range(20):
        instances.append((f"seed {seed}", *tiny_instance(seed)))
    orders, fleet = tiny_instance(1)
    orders.loc[:, ["scenario", "depot", "brand", "temp_requirement", "parking_constraint"]] = [
        "S1", "Peliyagoda", "Fresh", "chilled", "normal"]
    orders.loc[:, "district"] = ["Colombo", "Colombo", "Gampaha", "Gampaha", "Matara", "Puttalam"]
    orders.loc[:, "order_volume_m3"] = [2, 3, 4, 5, 6, 7]
    orders.loc[:, "order_weight_kg"] = [400, 600, 800, 1000, 1200, 1400]
    orders.loc[orders.index[0], "parking_constraint"] = "van_only"
    fleet.loc[~fleet.is_van, ["volume_cap_m3", "weight_cap_kg"]] = [7.0, 1600]
    instances.append(("three interchangeable trucks", orders, fleet))
    # Scenario belongs to vehicle capability: superficially identical trucks
    # serving different scenarios cannot safely share the mask-order symmetry.
    orders, fleet = tiny_instance(0)
    orders = orders.iloc[:2].copy()
    orders.loc[:, ["depot", "parking_constraint", "temp_requirement"]] = ["Peliyagoda", "normal", "chilled"]
    orders.loc[:, "scenario"] = ["S1", "S2"]
    orders.loc[:, "district"] = "Matara"
    orders.loc[:, "order_volume_m3"] = 10.0
    orders.loc[:, "order_weight_kg"] = 1000.0
    fleet.loc[~fleet.is_van, ["volume_cap_m3", "weight_cap_kg"]] = [12.0, 1900]
    fleet.loc[fleet.vehicle_id.eq("VEH006"), "scenario"] = "S2"
    instances.append(("different vehicle scenarios", orders, fleet))
    for name, orders, fleet in instances:
        priority, volume, explored = brute_force(orders, fleet)
        nodes += explored
        for volume_first, expected in ((False, priority), (True, volume)):
            allocation, certificate = solve(orders, fleet, volume_first=volume_first)
            actual = tuple(certificate["chilled_objective"])
            assert actual == expected, (name, volume_first, expected, actual, allocation)
            assert witness_key(orders, fleet, allocation, volume_first) == expected
    # Empty input and no available reefers are valid boundary instances.
    orders, fleet = tiny_instance(0)
    for empty_orders, empty_fleet in ((orders.iloc[:0], fleet), (orders, fleet.iloc[:0])):
        for mode in (False, True):
            _, certificate = solve(empty_orders, empty_fleet, volume_first=mode)
            assert tuple(certificate["chilled_objective"]) == (0, 0, 0, 0)
    print(f"Exact-search independent exhaustive checks PASSED: {len(instances)} instances, "
          f"{2*len(instances)} objectives, "
          f"4 empty cases, {nodes:,} oracle nodes.")


if __name__ == "__main__":
    main()
