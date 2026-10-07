"""Independent exhaustive stop-order oracle for the Phase 5 window DP.

These tests verify diagnostics, not additional Task 2B submission constraints.
Run from the repository root: python task2b/verify_timeline.py.
"""
import sys
from datetime import datetime
from itertools import permutations

import pandas as pd

sys.path.insert(0, ".")
from task2b.audit import sequence_trip, sequence_window, effective_window
from task2b.common import load_orders, load_travel, load_allowance


def oracle_clock(value):
    value = datetime.strptime(value, "%H:%M")
    return value.hour * 60 + value.minute


def oracle_windows(row):
    opening, closing = oracle_clock(row.window_open_time), oracle_clock(row.window_close_time)
    if row.parking_constraint == "mall_dock" and pd.notna(row.mall_window):
        mall_open, mall_close = row.mall_window.split("-")
        opening, closing = max(opening, oracle_clock(mall_open)), min(closing, oracle_clock(mall_close))
    return opening, closing


def exhaustive_finish(group, departure, travel, allowance, require_on_time):
    """Enumerate every permutation directly without any DP or production helpers."""
    best = None
    for rows in permutations(list(group.itertuples())):
        current = departure
        for position, row in enumerate(rows):
            district = travel[row.district]
            current += (district["depot_to_district_freeflow_min"] if position == 0
                        else district["inter_stop_freeflow_min"])
            opening, closing = oracle_windows(row)
            if require_on_time and current > closing:
                break
            current = max(current, opening) + allowance[(row.brand, row.dock_type)]
        else:
            best = current if best is None else min(best, current)
    return best


def check_witness(group, departure, result, travel, allowance, require_on_time):
    finish, records = result
    assert len(records) == len(group)
    assert {record["order_ref"] for record in records} == set(group.order_ref)
    assert len({record["order_ref"] for record in records}) == len(records)
    rows = {row.order_ref: row for row in group.itertuples()}
    current = departure
    for position, record in enumerate(records):
        row = rows[record["order_ref"]]
        district = travel[row.district]
        leg = (district["depot_to_district_freeflow_min"] if position == 0
               else district["inter_stop_freeflow_min"])
        arrival = current + leg
        opening, closing = oracle_windows(row)
        start = max(arrival, opening)
        end = start + allowance[(row.brand, row.dock_type)]
        assert record["sequence"] == position + 1
        assert record["departure_min"] == current and record["travel_min"] == leg
        assert record["arrival_min"] == arrival
        assert record["wait_min"] == start - arrival
        assert record["service_start_min"] == start and record["service_end_min"] == end
        assert record["early_arrival"] == (arrival < opening)
        assert record["arrival_late"] == (arrival > closing)
        assert record["finish_after_close"] == (end > closing)
        if require_on_time:
            assert arrival <= closing
        current = end
    assert current == finish


def main():
    orders, travel, allowance = load_orders(), load_travel(), load_allowance()
    groups = [orders[(orders.brand == "Fresh") & (orders.district == "Colombo")].iloc[:6],
              orders[(orders.brand == "Fresh") & (orders.district == "Gampaha")].iloc[:6],
              orders[(orders.brand == "Tech") & (orders.district == "Colombo")]]
    count = 0
    infeasible_cases = 0
    for group in groups:
        for departure in (210, 330, 450, 540, 660):
            for constrained in (True, False):
                expected = exhaustive_finish(group, departure, travel, allowance, constrained)
                actual = sequence_trip(group, departure, travel, allowance, constrained)
                assert (actual is None) == (expected is None), (departure, constrained)
                if actual is None:
                    infeasible_cases += 1
                else:
                    assert actual[0] == expected, (departure, constrained, expected, actual[0])
                    check_witness(group, departure, actual, travel, allowance, constrained)
                count += 1
    assert count == 30 and infeasible_cases > 0

    # Mall opening causes a wait; its effective window overrides wider outlet hours.
    mall = orders[orders.parking_constraint.eq("mall_dock") & orders.brand.eq("Tech")].iloc[:1].copy()
    mall.loc[:, ["window_open_time", "window_close_time"]] = ["09:00", "17:00"]
    row = next(mall.itertuples())
    assert effective_window(row) == (600, 720)
    result = sequence_trip(mall, 540, travel, allowance, True)
    assert result is not None
    check_witness(mall, 540, result, travel, allowance, True)
    record = result[1][0]
    assert record["arrival_min"] == 564 and record["wait_min"] == 36
    assert record["service_start_min"] == 600 and record["service_end_min"] == 655

    # An arrival exactly at closing is on time even if unloading finishes afterwards.
    single = groups[0].iloc[:1]
    row = next(single.itertuples())
    closing = oracle_windows(row)[1]
    outbound = travel[row.district]["depot_to_district_freeflow_min"]
    departure = closing - outbound
    result = sequence_trip(single, departure, travel, allowance, True)
    assert result is not None
    check_witness(single, departure, result, travel, allowance, True)
    assert not result[1][0]["arrival_late"] and result[1][0]["finish_after_close"]
    assert sequence_trip(single, departure + 1, travel, allowance, True) is None
    records, on_time = sequence_window([(1, single)], departure + 1, travel, allowance)
    assert not on_time and len(records) == 1 and records[0]["arrival_late"]

    # Only swapping the two Fresh trips gives on-time arrivals in this fixture.
    fixture = single.iloc[[0, 0]].copy().reset_index(drop=True)
    fixture.loc[:, "order_ref"] = ["TEST-A", "TEST-B"]
    fixture.loc[:, "outlet_id"] = ["OUT-A", "OUT-B"]
    fixture.loc[:, "district"] = ["Near", "Far"]
    fixture.loc[:, "dock_type"] = "rear_dock"
    fixture.loc[:, "parking_constraint"] = "normal"
    fixture.loc[:, "window_open_time"] = "03:30"
    fixture.loc[:, "window_close_time"] = ["04:20", "03:55"]
    fixture_travel = {"Near": {"depot_to_district_freeflow_min": 10, "inter_stop_freeflow_min": 5},
                      "Far": {"depot_to_district_freeflow_min": 20, "inter_stop_freeflow_min": 5}}
    records, on_time = sequence_window([(1, fixture.iloc[:1]), (2, fixture.iloc[1:])],
                                       210, fixture_travel, allowance)
    assert on_time
    assert [record["trip_id"] for record in records] == [2, 1]
    assert [record["arrival_min"] for record in records] == [230, 255]
    assert sequence_window([], 210, travel, allowance) == ([], True)
    print(f"Timeline independent exhaustive checks PASSED: {count} permutation-oracle cases "
          f"({infeasible_cases} infeasible); mall wait/intersection, closing boundary, "
          "late fallback, trip swapping and empty-window cases.")


if __name__ == "__main__":
    main()
