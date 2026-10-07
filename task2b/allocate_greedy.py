"""Phase 3: reproducible scarce-first baseline, separate ambient allocation."""
import sys

sys.path.insert(0, ".")
from task2b.common import (DATA_DIR, REPORT_DIR, FRESH_BUDGET_MIN, DAYTIME_BUDGET_MIN,
                           load_orders, load_fleet, load_travel, load_allowance, trip_time)
from task2b.analysis import individually_feasible
from task2b.policy import order_key, score
from task2b.feasibility import enumerate_trips, feasible_trip, make_submission, validate, official_check


def allocate_ambient(orders, fleet):
    """Fill ambient vans then trucks, by brand/district, allowing at most two trips."""
    travel, allowance = load_travel(), load_allowance()
    vehicles = sorted((v for v in fleet.itertuples() if v.available and not v.is_reefer),
                      key=lambda v: (not v.is_van, -v.volume_cap_m3, v.vehicle_id))
    allocation = {v.vehicle_id: [] for v in vehicles}
    budgets = {v.vehicle_id: {"Fresh": 0, "daytime": 0} for v in vehicles}
    ambient = orders[orders.temp_requirement.eq("ambient")]
    impossible = {r.Index for r in ambient.itertuples()
                  if not any(individually_feasible(r, v, travel, allowance) for v in vehicles)}
    remaining = set(ambient.index) - impossible
    # Allocate access-constrained orders before normal trucks. Every trip stays homogeneous.
    groups = sorted(ambient.groupby(["brand", "district", "parking_constraint"]).groups.items(),
                    key=lambda item: (item[0][2] != "van_only", item[0][0] != "Fresh", item[0]))
    for (brand, district, _), indices in groups:
        waiting = sorted(remaining.intersection(indices), key=lambda i: order_key(orders.loc[i]))
        while waiting:
            chosen = None
            for v in vehicles:
                if len(allocation[v.vehicle_id]) >= 2:
                    continue
                selected = []
                category = "Fresh" if brand == "Fresh" else "daytime"
                budget = FRESH_BUDGET_MIN if brand == "Fresh" else DAYTIME_BUDGET_MIN
                for i in waiting:
                    candidate = orders.loc[selected + [i]]
                    minutes = trip_time(district, brand, list(candidate.dock_type), travel, allowance)
                    if feasible_trip(candidate, v, travel, allowance) and budgets[v.vehicle_id][category] + minutes <= budget:
                        selected.append(i)
                if selected:
                    minutes = trip_time(district, brand, list(orders.loc[selected].dock_type), travel, allowance)
                    key = (len(selected), -int(v.is_van), -minutes, v.vehicle_id)
                    if chosen is None or key > chosen[0]:
                        chosen = (key, v, tuple(selected), minutes, category)
            if chosen is None:
                raise RuntimeError(f"Greedy ambient allocation stranded {waiting}; improve packing")
            _, v, selected, minutes, category = chosen
            allocation[v.vehicle_id].append(selected)
            budgets[v.vehicle_id][category] += minutes
            remaining.difference_update(selected)
            waiting = [i for i in waiting if i not in selected]
    assert not remaining
    assert impossible == {78}, impossible
    return {v: trips for v, trips in allocation.items() if trips}


def allocate_chilled(orders, fleet):
    travel, allowance = load_travel(), load_allowance()
    vehicles = sorted((v for v in fleet.itertuples() if v.available and v.is_reefer),
                      key=lambda v: (not v.is_van, -v.volume_cap_m3, v.vehicle_id))
    allocation, used = {}, 0
    for v in vehicles:
        trips = enumerate_trips(orders, v, travel, allowance)
        selected, minutes = [], 0
        for _ in range(2):
            candidates = [t for t in trips if not (t.mask & used) and minutes + t.minutes <= FRESH_BUDGET_MIN]
            if v.is_van and not selected:
                # Baseline reserves the van for constrained chilled orders if possible.
                access = [t for t in candidates if orders.loc[list(t.indices)].parking_constraint.eq("van_only").all()]
                candidates = access or candidates
            if not candidates:
                break
            chosen = max(candidates, key=lambda t: (t.score, t.indices))
            selected.append(chosen.indices)
            minutes += chosen.minutes
            used |= chosen.mask
        if selected:
            allocation[v.vehicle_id] = selected
    return allocation


def main():
    orders, fleet = load_orders(), load_fleet()
    allocation = allocate_ambient(orders, fleet) | allocate_chilled(orders, fleet)
    sub = make_submission(orders, allocation)
    minutes = validate(sub, orders, fleet)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "greedy_submission.csv"
    sub.to_csv(path, index=False)
    checked = official_check(path)
    served = orders.loc[sub.decision.eq("served")]
    output = (f"Greedy scarce-first baseline\nScore (protected, chilled litres, orders, outlets, -minutes): "
              f"{score(served.itertuples(), minutes)}\nDeferred orders: "
              f"{sub.loc[sub.decision.eq('deferred'), 'order_ref'].tolist()}\n{checked}\n")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "greedy_summary.txt").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
