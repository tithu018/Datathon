"""Task 2B Phase 0 smoke check: data in place, formula matches the checker, checker runs.

1. Input checks: row counts, unique keys, template rows match the scenario, every
   district and (brand, dock_type) has a reference row, fleet joins vehicles.csv.
2. trip_time() reproduces both worked examples in the Challenge Booklet (101 and 112
   min) and equals check_allocation.trip_time() for every district x brand x dock.
3. An all-deferred submission (trivially feasible) passes check_allocation.py, and a
   deliberately broken one (chilled order on an ambient truck) fails it.

Run from the repo root. Writes task2b/reports/phase0_smoke_output.txt.
check_allocation.py looks for the data under data/ (see data/README.md).
"""
import io
import subprocess
import sys
from contextlib import redirect_stdout

import pandas as pd

sys.path.insert(0, ".")
import check_allocation  # noqa: E402  (organisers' checker, repo root)
from task2b.common import (CHECKER, DATA_DIR, REPORT_DIR, TEMPLATE, load_allowance,  # noqa: E402
                           load_fleet, load_orders, load_travel, trip_time)

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail != '' else ''}")
    if not ok:
        failures.append(name)


def run_checker(sub, name):
    """Write sub to task2b/data/<name> and run the official checker on it."""
    path = DATA_DIR / name
    sub.to_csv(path, index=False)
    r = subprocess.run([sys.executable, str(CHECKER), str(path)], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    orders, fleet = load_orders(), load_fleet()
    travel, allowance = load_travel(), load_allowance()
    template = pd.read_csv(TEMPLATE)

    print("1. Inputs")
    check("85 orders, order_ref unique", len(orders) == 85 and orders.order_ref.is_unique, len(orders))
    check("template rows == scenario rows, same order",
          template.order_ref.tolist() == orders.order_ref.tolist()
          and template.outlet_id.tolist() == orders.outlet_id.tolist(), len(template))
    check("scenario is always S1", orders.scenario.eq("S1").all())
    check("all orders at Peliyagoda", orders.depot.eq("Peliyagoda").all(),
          orders.depot.value_counts().to_dict())
    check("no missing order sizes",
          orders[["order_units", "order_weight_kg", "order_volume_m3"]].notna().all().all())
    check("every district has a district_travel row", set(orders.district) <= set(travel),
          sorted(set(orders.district) - set(travel)) or "")
    check("every (brand, dock_type) has an allowance",
          set(zip(orders.brand, orders.dock_type)) <= set(allowance))
    check("fleet joins vehicles.csv", fleet.type.notna().all(), f"{len(fleet)} vehicles")
    avail = fleet[fleet.available]
    check("28 available, 10 in_workshop", len(avail) == 28 and (~fleet.available).sum() == 10,
          fleet.status.value_counts().to_dict())
    check("all available vehicles are Peliyagoda", avail.depot.eq("Peliyagoda").all())

    print("\n2. Trip-time formula")
    t1 = trip_time("Gampaha", "Fresh", ["rear_dock", "rear_dock", "street"], travel, allowance)
    check("booklet example: Fresh Gampaha rear, rear, street = 101", t1 == 101, t1)
    t2 = trip_time("Colombo", "Fresh", ["street"] * 4, travel, allowance)
    check("booklet example: Fresh Colombo 4 x street = 112", t2 == 112, t2)
    check("101 + 112 = 213 of 270 Fresh minutes", t1 + t2 == 213, t1 + t2)
    docks = ["rear_dock", "street", "mall_bay"]
    diffs = [(d, b) for d in travel for b in ("Fresh", "Style", "Tech")
             if trip_time(d, b, docks, travel, allowance)
             != check_allocation.trip_time(d, b, docks, travel, allowance)]
    check("equals check_allocation.trip_time for every district x brand", not diffs, diffs or "")

    print("\n3. Official checker")
    sub = template.copy()
    sub["decision"], sub["vehicle_id"], sub["trip_id"] = "deferred", "", ""
    code, out = run_checker(sub, "smoke_all_deferred.csv")
    print("    " + out.replace("\n", "\n    "))
    check("all-deferred submission passes", code == 0 and "PASSED" in out)

    chilled = orders.index[orders.temp_requirement.eq("chilled")
                           & orders.parking_constraint.eq("normal")][0]
    ambient_truck = avail[~avail.is_reefer & ~avail.is_van].vehicle_id.iloc[0]
    bad = sub.copy()
    bad.loc[chilled, ["decision", "vehicle_id", "trip_id"]] = ["served", ambient_truck, "1"]
    code, out = run_checker(bad, "smoke_broken.csv")
    print("    " + out.replace("\n", "\n    "))
    check(f"chilled {orders.order_ref[chilled]} on ambient {ambient_truck} is rejected",
          code == 1 and "non-refrigerated" in out)

    print(f"\nSmoke check: {'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main()
    text = buf.getvalue()
    print(text, end="")
    (REPORT_DIR / "phase0_smoke_output.txt").write_text(text, encoding="utf-8")
    sys.exit(code)
