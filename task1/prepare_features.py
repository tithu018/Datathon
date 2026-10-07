import pandas as pd

train = pd.read_csv(r"task1\task1_training_labels.csv")
test_orders = pd.read_csv(r"Test Data\task1_test_inputs.csv")
test_legs = pd.read_csv(r"Test Data\route_legs_test.csv")

# Merge test orders with their matching planned route leg
test = test_orders.merge(
    test_legs,
    left_on=["route_id", "seq_in_route"],
    right_on=["route_id", "seq"],
    how="left",
    suffixes=("_order", "_leg")
)

def time_to_minutes(series):
    t = pd.to_datetime(series, format="%H:%M", errors="coerce")
    return t.dt.hour * 60 + t.dt.minute

# Planned-time features only
for df in [train, test]:
    if "planned_arrival_time_order" in df.columns:
        df["planned_arrival_min"] = time_to_minutes(df["planned_arrival_time_order"])
    elif "planned_arrival_time" in df.columns:
        df["planned_arrival_min"] = time_to_minutes(df["planned_arrival_time"])

    if "planned_depart_time" in df.columns:
        df["planned_depart_min"] = time_to_minutes(df["planned_depart_time"])

    if "window_open_time" in df.columns:
        df["window_open_min_feature"] = time_to_minutes(df["window_open_time"])

    if "window_close_time" in df.columns:
        df["window_close_min_feature"] = time_to_minutes(df["window_close_time"])

# Candidate features that should exist in both train/test
candidate_features = [
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

features = [c for c in candidate_features if c in train.columns and c in test.columns]

print("Usable features:")
for c in features:
    print("-", c)

print("\nTrain rows:", len(train))
print("Test rows:", len(test))
print("Matched test route legs:", test["distance_km"].notna().sum(), "/", len(test))

train[features + ["service_minutes", "late"]].to_csv(
    r"task1\train_features.csv", index=False
)

test[["delivery_id"] + features].to_csv(
    r"task1\test_features.csv", index=False
)

print("\nSaved:")
print("task1/train_features.csv")
print("task1/test_features.csv")