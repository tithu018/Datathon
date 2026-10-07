"""Phase 2: explicit lexicographic priority; volume in integer litres."""
import sys

sys.path.insert(0, ".")
from task2b.common import REPORT_DIR, load_orders


def protected(order):
    """Flags belong to order rows; OR avoids double-counting both indicators."""
    return bool(order.deferred_yesterday == 1 or order.days_since_last_served >= 3)


def volume_units(order):
    return int(round(order.order_volume_m3 * 1000))


def score(rows, minutes=0):
    """Maximise this tuple; never exchange a protected order for more volume."""
    rows = list(rows)
    return (sum(protected(r) for r in rows),
            sum(volume_units(r) for r in rows if r.temp_requirement == "chilled"),
            len(rows), len({r.outlet_id for r in rows}), -int(round(minutes)))


def order_key(order):
    """Deterministic greedy ordering, distinct from scoring complete plans."""
    return (-int(protected(order)), -volume_units(order), order.order_ref)


def main():
    rows = list(load_orders().itertuples())
    urgent = next(r for r in rows if protected(r) and r.temp_requirement == "chilled")
    ordinary = [r for r in rows if not protected(r) and r.temp_requirement == "chilled"]
    assert score([urgent]) > score(ordinary)  # urgency dominates any volume gain
    assert score([ordinary[0]], 10) > score([ordinary[0]], 11)
    assert score([]) == (0, 0, 0, 0, 0)
    assert sum(protected(r) for r in rows) == 10
    output = """Task 2B priority policy (lexicographic, maximised)
1. Number of served protected order rows: deferred_yesterday=1 OR days_since_last_served>=3.
   Both flags together count once. Flags differ between ambient/chilled rows at the same outlet;
   honour the supplied row flags, and audit outlet coverage separately.
2. Chilled volume, in integer units of 0.001 m3 (avoids floating-point tie breaks).
3. Number of orders served.
4. Number of distinct outlets served.
5. Negative published vehicle-minutes (no return or window-waiting time).
Exact search optimises the chilled subproblem with a fixed, independently feasible ambient allocation.
Ambient orders use ambient vehicles while chilled demand remains unserved.
An ambient order remains eligible when its paired chilled order is deferred.
Protected orders are prioritised wherever jointly feasible; no unsupported guarantee of serving every order.
Stable vehicle/order identifiers break exact score ties for reproducibility.
Checks: urgency dominates volume; time breaks otherwise-equal plans; flags count once; empty score; PASSED.
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "priority_policy.txt").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
