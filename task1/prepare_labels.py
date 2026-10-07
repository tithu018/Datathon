import pandas as pd

deliveries = pd.read_csv(r"Training Data\deliveries_train.csv")
legs = pd.read_csv(r"Training Data\route_legs_train.csv")

# Only orders that actually have a route
deliveries = deliveries[
    deliveries["route_id"].notna() &
    deliveries["seq_in_route"].notna()
].copy()

# Join each order to its matching route leg
df = deliveries.merge(
    legs,
    left_on=["route_id", "seq_in_route"],
    right_on=["route_id", "seq"],
    how="inner",
    suffixes=("_order", "_leg")
)

def time_to_minutes(t):
    h, m = map(int, str(t).split(":"))
    return h * 60 + m

df["arrival_min"] = df["arrival_time"].apply(time_to_minutes)
df["leave_min"] = df["leave_outlet_time"].apply(time_to_minutes)
df["window_open_min"] = df["window_open_time"].apply(time_to_minutes)
df["window_close_min"] = df["window_close_time"].apply(time_to_minutes)

# If vehicle arrives early, service starts only when outlet opens
df["service_start_min"] = df[["arrival_min", "window_open_min"]].max(axis=1)

df["service_minutes"] = df["leave_min"] - df["service_start_min"]

# Late = actual arrival after window closes
df["late"] = (df["arrival_min"] > df["window_close_min"]).astype(int)

# Basic checks
print("Training rows:", len(df))
print("Negative service times:", (df["service_minutes"] < 0).sum())
print("Missing service times:", df["service_minutes"].isna().sum())
print("Late rate:", round(df["late"].mean(), 4))

df.to_csv(r"task1\task1_training_labels.csv", index=False)

print("Saved: task1/task1_training_labels.csv")