import pandas as pd

from feature_groups import add_feature_groups
from features import ALL_CANDIDATES, FEATURES, assert_no_leakage

train = pd.read_csv(r"task1\task1_training_labels.csv")
test_orders = pd.read_csv(r"Test Data\task1_test_inputs.csv")
test_legs = pd.read_csv(r"Test Data\route_legs_test.csv")
train_legs = pd.read_csv(r"Training Data\route_legs_train.csv")

# Merge test orders with their matching planned route leg
test = test_orders.merge(
    test_legs,
    left_on=["route_id", "seq_in_route"],
    right_on=["route_id", "seq"],
    how="left",
    suffixes=("_order", "_leg")
)

# Phase 6 fix: the order and the route leg both carry these columns, so the merge
# renames them to <col>_order / <col>_leg and the plain names no longer exist. The
# original candidate list silently dropped them. They are identical in both files
# (task1/label_audit.py), so check that and restore the plain column name.
SHARED_COLUMNS = ["brand", "district", "depot", "vehicle_type", "vehicle_temp", "planned_arrival_time"]
for df in [train, test]:
    for c in SHARED_COLUMNS:
        assert (df[f"{c}_order"] == df[f"{c}_leg"]).all(), f"{c}: order and route leg disagree"
        df[c] = df[f"{c}_order"]

def time_to_minutes(series):
    t = pd.to_datetime(series, format="%H:%M", errors="coerce")
    assert t.notna().all(), "unparseable HH:MM time"
    return t.dt.hour * 60 + t.dt.minute

# Planned-time features only
for df in [train, test]:
    df["planned_arrival_min"] = time_to_minutes(df["planned_arrival_time"])
    df["planned_depart_min"] = time_to_minutes(df["planned_depart_time"])
    df["window_open_min_feature"] = time_to_minutes(df["window_open_time"])
    df["window_close_min_feature"] = time_to_minutes(df["window_close_time"])

# Phase 7 candidate groups (planned information only; task1/feature_groups.py)
train = add_feature_groups(train, train_legs)
test = add_feature_groups(test, test_legs)

# Explicit feature lists (task1/features.py): every column must exist in train and test
assert_no_leakage(ALL_CANDIDATES)
missing = {name: [c for c in ALL_CANDIDATES if c not in df.columns] for name, df in [("train", train), ("test", test)]}
assert not any(missing.values()), f"feature columns missing: {missing}"
assert test[ALL_CANDIDATES].notna().all().all(), "missing feature values in test"
assert train[ALL_CANDIDATES].notna().all().all(), "missing feature values in train"

print(f"Model features ({len(FEATURES)}):", ", ".join(FEATURES))
print(f"Candidate columns written ({len(ALL_CANDIDATES)}):", ", ".join(c for c in ALL_CANDIDATES if c not in FEATURES))

print("\nTrain rows:", len(train))
print("Test rows:", len(test))
print("Matched test route legs:", test["distance_km"].notna().sum(), "/", len(test))

# order_date and delivery_id are kept for the time-based split and traceability, not as features
train[["delivery_id", "order_date"] + ALL_CANDIDATES + ["service_minutes", "late"]].to_csv(
    r"task1\train_features.csv", index=False
)

test[["delivery_id"] + ALL_CANDIDATES].to_csv(
    r"task1\test_features.csv", index=False
)

print("\nSaved:")
print("task1/train_features.csv")
print("task1/test_features.csv")
