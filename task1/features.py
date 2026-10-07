"""Task 1 model features (explicit lists; prepare_features.py and train_model.py assert
that every column exists, so nothing can be dropped silently).

BASE_FEATURES are the Phase 6 features. FEATURE_GROUPS are the Phase 7 candidate
groups (task1/feature_groups.py), all known at planning time. FEATURES is the model's
final list: BASE_FEATURES plus the groups kept in Phase 7 (KEPT_GROUPS).
"""

BASE_FEATURES = [
    "brand",
    "district",
    "depot",
    "temp_requirement",
    "order_units",
    "order_weight_kg",
    "order_volume_m3",
    "vehicle_type",
    "vehicle_temp",
    "distance_km",
    "planned_travel_duration_min",
    "planned_arrival_min",
    "planned_depart_min",
    "window_open_min_feature",
    "window_close_min_feature",
    "monsoon",
    "dow",
]

FEATURE_GROUPS = {
    "outlet": ["dock_type", "parking_constraint", "mall_window_open_min", "mall_window_close_min"],
    "service_allowance": ["service_allowance_min"],
    "planned_slack": ["slack_to_close_min", "slack_from_open_min", "planned_dwell_min", "is_last_stop"],
    "route_context": ["seq", "n_stops", "cum_planned_travel_min", "planned_route_start_min"],
    "calendar": ["is_payday", "festival_ramp", "is_holiday"],
    "traffic": ["speed_index"],
    "road_conditions": ["disruption_index"],  # added at the user's decision (Phase 7)
}

# Measured in Phase 7 as a separate, labelled extra; never kept without the user's decision
EXTRA_CANDIDATES = {"unplanned_wait": ["cum_unplanned_wait_before"]}

KEPT_GROUPS = ["outlet", "route_context", "calendar", "traffic", "road_conditions"]  # Phase 7 (reports/phase7_feature_groups.csv)

FEATURES = BASE_FEATURES + [c for g in KEPT_GROUPS for c in FEATURE_GROUPS[g]]

CATEGORICAL = ["brand", "district", "depot", "temp_requirement", "vehicle_type", "vehicle_temp",
               "dock_type", "parking_constraint"]

# Columns that exist only after delivery (training route records): never features
LEAKAGE = {"actual_depart_time", "actual_travel_duration_min", "arrival_time", "leave_outlet_time",
           "arrival_min", "leave_min", "service_start_min", "service_minutes", "late"}


def assert_no_leakage(features):
    bad = [c for c in features if c in LEAKAGE or c.startswith("actual_")]
    assert not bad, f"leakage: post-delivery columns in the feature list: {bad}"


ALL_CANDIDATES = BASE_FEATURES + [c for groups in (FEATURE_GROUPS, EXTRA_CANDIDATES)
                                   for cols in groups.values() for c in cols]
assert_no_leakage(ALL_CANDIDATES)
