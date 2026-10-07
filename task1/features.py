"""Task 1 model features (explicit list; prepare_features.py and train_model.py assert
that every column exists, so nothing can be dropped silently)."""

FEATURES = [
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

CATEGORICAL = ["brand", "district", "depot", "temp_requirement", "vehicle_type", "vehicle_temp"]
