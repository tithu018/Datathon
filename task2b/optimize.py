"""Phase 4: solver-free exact chilled set packing with branch-and-bound.

Enumerate every feasible one/two-trip schedule per reefer. Select one schedule
per vehicle with disjoint order masks. Bounds only relax constraints, so pruning
cannot remove an improving plan. Ambient allocation is fixed and separate.
"""
import json
import sys
from dataclasses import dataclass

sys.path.insert(0, ".")
from task2b.common import (DATA_DIR, REPORT_DIR, FRESH_BUDGET_MIN, load_orders,
                           load_fleet, load_travel, load_allowance)
from task2b.policy import protected, volume_units, score
from task2b.feasibility import enumerate_trips, make_submission, validate, official_check
from task2b.allocate_greedy import allocate_ambient


@dataclass(frozen=True)
class Schedule:
    mask: int
    urgent: int
    volume: int
    count: int
    minutes: int
    trips: tuple

    def key(self, volume_first=False):
        if volume_first:
            return (self.volume, self.urgent, self.count, -self.minutes)
        return (self.urgent, self.volume, self.count, -self.minutes)


def schedules_for(trips):
    """Same served mask: retain minimum-time partition; other objectives equal."""
    schedules = {0: Schedule(0, 0, 0, 0, 0, ())}

    def put(a, b=None):
        mask = a.mask | (b.mask if b else 0)
        minutes = a.minutes + (b.minutes if b else 0)
        partition = tuple(sorted((a.indices, b.indices) if b else (a.indices,)))
        candidate = Schedule(mask, a.urgent + (b.urgent if b else 0),
                             a.volume + (b.volume if b else 0), mask.bit_count(), minutes, partition)
        existing = schedules.get(mask)
        if existing is None or (minutes, partition) < (existing.minutes, existing.trips):
            schedules[mask] = candidate

    for i, a in enumerate(trips):
        put(a)
        for b in trips[i + 1:]:
            if not a.mask & b.mask and a.minutes + b.minutes <= FRESH_BUDGET_MIN:
                put(a, b)
    return list(schedules.values())


def solve(orders, fleet, volume_first=False):
    travel, allowance = load_travel(), load_allowance()
    vehicles = [v for v in fleet.itertuples() if v.available and v.is_reefer]
    generated = {}
    trip_counts = {}
    for v in vehicles:
        trips = enumerate_trips(orders, v, travel, allowance)
        trip_counts[v.vehicle_id] = len(trips)
        generated[v.vehicle_id] = sorted(schedules_for(trips), key=lambda c: (c.key(volume_first), c.trips), reverse=True)
    # Leave the smallest schedule set last: its best conflict-free schedule is a
    # direct scan. Trucks with larger capacities are considered first.
    vehicles.sort(key=lambda v: (v.is_van, -v.volume_cap_m3, v.vehicle_id))
    options = [generated[v.vehicle_id] for v in vehicles]
    chilly = list(orders[orders.temp_requirement.eq("chilled")].itertuples())
    urgent_mask = sum(1 << r.Index for r in chilly if protected(r))
    all_mask = sum(1 << r.Index for r in chilly)
    values = {r.Index: volume_units(r) for r in chilly}
    suffix_masks = [0] * (len(vehicles) + 1)
    for i in range(len(vehicles) - 1, -1, -1):
        suffix_masks[i] = suffix_masks[i + 1]
        for c in options[i]:
            suffix_masks[i] |= c.mask
    best_key = (-1, -1, -1, -10**9)
    best_chosen = ()
    nodes = pruned = 0

    def ordered_key(urgent, volume, count, minutes):
        return (volume, urgent, count, -minutes) if volume_first else (urgent, volume, count, -minutes)

    def remaining_volume(mask):
        return sum(values[i] for i in values if mask & (1 << i))

    def relaxed_best(sets, urgent_limit):
        # Multiple-choice knapsack over protected counts, relaxing conflicts
        # between vehicles. Keep volume/count/time together for tighter bounds.
        states = {0: (0, 0, 0)}
        for opts in sets:
            categories = {}
            for c in opts:
                value = (c.volume, c.count, -c.minutes)
                if value > categories.get(c.urgent, (-1, -1, -10**9)):
                    categories[c.urgent] = value
            updated = {}
            for previous, value in states.items():
                for extra, addition in categories.items():
                    u = previous + extra
                    if u <= urgent_limit:
                        candidate = tuple(a + b for a, b in zip(value, addition))
                        if candidate > updated.get(u, (-1, -1, -10**9)):
                            updated[u] = candidate
            states = updated
        return states

    static_relaxed = [relaxed_best(options[d:], urgent_mask.bit_count()) for d in range(len(vehicles) + 1)]

    def dfs(depth, used, urgent, volume, count, minutes, chosen):
        nonlocal best_key, best_chosen, nodes, pruned
        nodes += 1
        if depth == len(vehicles):
            candidate_key = ordered_key(urgent, volume, count, minutes)
            if candidate_key > best_key:
                best_key, best_chosen = candidate_key, chosen
            return
        # Remaining mask and the per-vehicle compatible schedule maxima are
        # independent relaxations. Each component bounds its true completion.
        possible = suffix_masks[depth] & ~used & all_mask
        max_urgent = urgent + (possible & urgent_mask).bit_count()
        max_volume = volume + remaining_volume(possible)
        compatible_sets = []
        for opts in options[depth:]:
            eligible = [c for c in opts if not c.mask & used]
            compatible_sets.append(eligible)
        relaxed = relaxed_best(compatible_sets, (possible & urgent_mask).bit_count())
        upper = max(ordered_key(urgent + u, volume + value[0],
                                count + value[1], minutes - value[2])
                    for u, value in relaxed.items())
        upper = min(upper, ordered_key(max_urgent, max_volume, count + possible.bit_count(), minutes))
        if upper <= best_key:
            pruned += 1
            return
        candidates = compatible_sets[0]
        if depth == len(vehicles) - 1:
            # Additive tuple objectives: first eligible candidate is optimal.
            c = candidates[0]
            dfs(depth + 1, used | c.mask, urgent + c.urgent, volume + c.volume,
                count + c.count, minutes + c.minutes, chosen + (c,))
            return
        for c in candidates:
            # Equal-capability vehicles are interchangeable in this objective.
            # Canonical mask order removes symmetric assignments, not scores.
            if depth and (vehicles[depth].scenario, vehicles[depth].type, vehicles[depth].temp, vehicles[depth].depot,
                          vehicles[depth].volume_cap_m3, vehicles[depth].weight_cap_kg) == (
                          vehicles[depth-1].scenario, vehicles[depth-1].type, vehicles[depth-1].temp, vehicles[depth-1].depot,
                          vehicles[depth-1].volume_cap_m3, vehicles[depth-1].weight_cap_kg):
                if c.mask > chosen[-1].mask:
                    continue
            # Cheap, safe mask-only bound before expensive next-node filtering.
            left = suffix_masks[depth + 1] & ~(used | c.mask) & all_mask
            possible_urgent = (left & urgent_mask).bit_count()
            left_volume = remaining_volume(left)
            next_upper = max(ordered_key(urgent + c.urgent + u,
                                     volume + c.volume + value[0],
                                     count + c.count + value[1], minutes + c.minutes - value[2])
                             for u, value in static_relaxed[depth + 1].items() if u <= possible_urgent)
            next_upper = min(next_upper, ordered_key(urgent + c.urgent + possible_urgent,
                                     volume + c.volume + left_volume,
                                     count + c.count + left.bit_count(), minutes + c.minutes))
            if next_upper <= best_key:
                pruned += 1
                continue
            dfs(depth + 1, used | c.mask, urgent + c.urgent, volume + c.volume,
                count + c.count, minutes + c.minutes, chosen + (c,))

    dfs(0, 0, 0, 0, 0, 0, ())
    allocation = {v.vehicle_id: list(c.trips) for v, c in zip(vehicles, best_chosen) if c.trips}
    certificate = {
        "objective_order": "volume_first" if volume_first else "protected_first",
        "chilled_objective": list(best_key), "trip_counts": trip_counts,
        "schedule_counts": {v: len(opts) for v, opts in generated.items()},
        "nodes": nodes, "pruned": pruned, "search_complete": True,
        "scope": "All feasible chilled assignments under published budgets, no ambient loads on reefers; ambient assignment fixed. Outlet coverage constant because every chilled outlet has served ambient order.",
    }
    return allocation, certificate


def main():
    orders, fleet = load_orders(), load_fleet()
    ambient = allocate_ambient(orders, fleet)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, volume_first in (("optimized", False), ("volume_first", True)):
        allocation, certificate = solve(orders, fleet, volume_first)
        sub = make_submission(orders, ambient | allocation)
        minutes = validate(sub, orders, fleet)
        path = DATA_DIR / f"{name}_submission.csv"
        sub.to_csv(path, index=False)
        checked = official_check(path)
        served = orders.loc[sub.decision.eq("served")]
        certificate["full_score"] = list(score(served.itertuples(), minutes))
        certificate["checker"] = checked
        results[name] = certificate
        print(f"{name}: {certificate['full_score']}; complete search {certificate['nodes']} nodes")
    greedy = __import__("pandas").read_csv(DATA_DIR / "greedy_submission.csv")
    greedy_score = score(orders.loc[greedy.decision.eq("served")].itertuples(), validate(greedy, orders, fleet))
    assert tuple(results["optimized"]["full_score"]) >= greedy_score
    results["greedy_score"] = list(greedy_score)
    results["chilled_volume_gain_m3"] = (results["optimized"]["full_score"][1] - greedy_score[1]) / 1000
    results["priority_volume_cost_m3"] = (results["volume_first"]["full_score"][1] - results["optimized"]["full_score"][1]) / 1000
    (REPORT_DIR / "optimization_summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("Exact optimisation and official checks PASSED")


if __name__ == "__main__":
    main()
