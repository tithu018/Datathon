"""Task 2B shared paths, loaders and the planning trip-time formula.

Task 2B: allocate the Peliyagoda fleet on one peak day (scenario S1). Every order is
served or deferred; served orders get a vehicle_id and a trip_id (1 or 2).

The constants and trip_time() mirror check_allocation.py (the organisers' checker) so
our allocator and the official check can never disagree.

Run scripts from the repo root.
"""
from pathlib import Path

import pandas as pd

SCENARIOS = Path("Test Data") / "task2b_peak_day_scenarios.csv"
FLEET = Path("Test Data") / "task2b_peak_day_fleet.csv"
VEHICLES = Path("General Data") / "vehicles.csv"
DISTRICT_TRAVEL = Path("General Data") / "district_travel.csv"
SERVICE_ALLOWANCE = Path("General Data") / "service_allowance.csv"
TEMPLATE = Path("Submission Templates") / "submission_task2b.csv"
CHECKER = Path("check_allocation.py")

DATA_DIR = Path("task2b") / "data"        # generated, gitignored
REPORT_DIR = Path("task2b") / "reports"
SUBMISSION = Path("outputs") / "submission_task2b.csv"

# Daily time budgets per vehicle (Challenge Booklet, Task 2B), in minutes.
FRESH_BUDGET_MIN = 270      # Fresh trips, 03:30-08:00
DAYTIME_BUDGET_MIN = 480    # Style and Tech trips combined
MAX_TRIPS = 2

SUBMISSION_COLS = ["scenario", "order_ref", "outlet_id", "decision", "vehicle_id", "trip_id"]


def load_orders():
    """One row per S1 order (task2b_peak_day_scenarios.csv)."""
    return pd.read_csv(SCENARIOS)


def load_fleet():
    """Every S1 vehicle joined to vehicles.csv, with an `available` flag."""
    fleet = pd.read_csv(FLEET)
    veh = pd.read_csv(VEHICLES)
    fleet = fleet.merge(veh, on="vehicle_id", how="left", validate="one_to_one")
    fleet["available"] = fleet.status.eq("available")
    fleet["is_reefer"] = fleet.temp.eq("reefer")
    fleet["is_van"] = fleet.type.eq("van")
    return fleet


def load_travel():
    """district -> {depot_to_district_freeflow_min, inter_stop_freeflow_min, ...}."""
    return pd.read_csv(DISTRICT_TRAVEL).set_index("district").to_dict("index")


def load_allowance():
    """(brand, dock_type) -> service_allowance_min."""
    al = pd.read_csv(SERVICE_ALLOWANCE)
    return {(r.brand, r.dock_type): r.service_allowance_min for r in al.itertuples()}


def trip_time(district, brand, docks, travel, allowance):
    """Planned trip minutes: outbound + inter-stop + handling (no return leg).

    Same formula as check_allocation.trip_time(): each order is one stop, so an
    outlet with a chilled and an ambient order on the same trip counts twice.
    """
    n = len(docks)
    if n == 0:
        return 0.0
    d = travel[district]
    return (d["depot_to_district_freeflow_min"]
            + (n - 1) * d["inter_stop_freeflow_min"]
            + sum(allowance[(brand, dk)] for dk in docks))
