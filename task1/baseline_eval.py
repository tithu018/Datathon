"""Phase 0 baseline: score the ORIGINAL Task 1 models (unchanged features,
preprocessing and hyperparameters, copied from train_model.py) on
  1. the original random 80/20 split (random_state=42, stratified on late), and
  2. a time-based test: train on order_date < 2026-01-03, test on the last
     6 weeks of training orders (order_date >= 2026-01-03).
Writes task1/reports/baseline_metrics.json. Run from the repo root after
task1/prepare_labels.py and task1/prepare_features.py.
"""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (brier_score_loss, log_loss, mean_absolute_error,
                             mean_squared_error, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
TIME_CUTOFF = "2026-01-03"
TASK1 = Path("task1")
OUT = TASK1 / "reports" / "baseline_metrics.json"

df = pd.read_csv(TASK1 / "train_features.csv")
labels = pd.read_csv(TASK1 / "task1_training_labels.csv", usecols=["order_date", "service_minutes", "late"])

# train_features.csv is written row-for-row from task1_training_labels.csv
assert len(df) == len(labels)
assert (df["service_minutes"].values == labels["service_minutes"].values).all()
assert (df["late"].values == labels["late"].values).all()
order_date = pd.to_datetime(labels["order_date"])

# The 12 features that survived the original (eaf7297) merge. Selected explicitly so this
# baseline still reproduces after Phase 6 added columns to train_features.csv.
ORIGINAL_FEATURES = ["temp_requirement", "order_units", "order_weight_kg", "order_volume_m3", "distance_km",
                     "planned_travel_duration_min", "planned_arrival_min", "planned_depart_min",
                     "window_open_min_feature", "window_close_min_feature", "monsoon", "dow"]
X = df[ORIGINAL_FEATURES]
y_service = df["service_minutes"]
y_late = df["late"]

# Same column split as train_model.py (object/str columns are categorical)
categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()


def make_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])


def make_models():
    service = Pipeline([("prep", make_preprocessor()), ("model", RandomForestRegressor(
        n_estimators=300, random_state=SEED, n_jobs=-1, min_samples_leaf=2))])
    late = Pipeline([("prep", make_preprocessor()), ("model", RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1, min_samples_leaf=2, class_weight="balanced"))])
    return service, late


def evaluate(train_idx, test_idx):
    service, late = make_models()
    service.fit(X.loc[train_idx], y_service.loc[train_idx])
    late.fit(X.loc[train_idx], y_late.loc[train_idx])
    s_pred = service.predict(X.loc[test_idx])
    p = late.predict_proba(X.loc[test_idx])[:, 1]
    ys, yl = y_service.loc[test_idx], y_late.loc[test_idx]
    return {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_late_rate": round(float(yl.mean()), 4),
        "service_mae": round(float(mean_absolute_error(ys, s_pred)), 4),
        "service_rmse": round(float(np.sqrt(mean_squared_error(ys, s_pred))), 4),
        "late_roc_auc": round(float(roc_auc_score(yl, p)), 4),
        "late_log_loss": round(float(log_loss(yl, np.clip(p, 1e-15, 1 - 1e-15))), 4),
        "late_brier": round(float(brier_score_loss(yl, p)), 4),
        "late_share_exact_0": round(float((p == 0).mean()), 4),
        "late_share_exact_1": round(float((p == 1).mean()), 4),
    }


# 1. Original random split (identical call to train_model.py)
rand_train, rand_test = train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=y_late)
random_metrics = evaluate(rand_train, rand_test)
print("Random split:", random_metrics)

# 2. Time-based test
time_test_mask = order_date >= pd.Timestamp(TIME_CUTOFF)
time_metrics = evaluate(df.index[~time_test_mask], df.index[time_test_mask])
time_metrics["train_order_dates"] = [str(order_date[~time_test_mask].min().date()), str(order_date[~time_test_mask].max().date())]
time_metrics["test_order_dates"] = [str(order_date[time_test_mask].min().date()), str(order_date[time_test_mask].max().date())]
print("Time-based test:", time_metrics)

result = {
    "description": "Original Task 1 pipeline (commit eaf7297), unchanged features and models",
    "features": X.columns.tolist(),
    "random_split": random_metrics,
    "time_based": {"cutoff_order_date": TIME_CUTOFF, **time_metrics},
    "notes": "log_loss computed with probabilities clipped to [1e-15, 1-1e-15]",
    "environment": {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    },
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2))
print("Saved:", OUT)
