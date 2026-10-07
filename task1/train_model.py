import json
from pathlib import Path

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from evaluation import SEED, TIME_CUTOFF, metrics, random_split, time_split
from features import CATEGORICAL, FEATURES

REPORTS = Path("task1") / "reports"
METRICS_PATH = REPORTS / "metrics_phase6.json"

df = pd.read_csv(r"task1\train_features.csv")

# Explicit features: fail loudly if any is missing (Phase 6 fix for silently dropped columns)
missing = [c for c in FEATURES if c not in df.columns]
assert not missing, f"feature columns missing from train_features.csv: {missing}"

X = df[FEATURES]
y_service = df["service_minutes"]
y_late = df["late"]

categorical_cols = [c for c in FEATURES if c in CATEGORICAL]
numeric_cols = [c for c in FEATURES if c not in CATEGORICAL]


def make_models():
    """Same preprocessing and Random Forest settings as the original pipeline (eaf7297)."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ]
    )
    service_model = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, min_samples_leaf=2)),
    ])
    late_model = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, min_samples_leaf=2,
                                         class_weight="balanced")),
    ])
    return service_model, late_model


def fit_and_score(train_idx, test_idx):
    service_model, late_model = make_models()
    service_model.fit(X.loc[train_idx], y_service.loc[train_idx])
    late_model.fit(X.loc[train_idx], y_late.loc[train_idx])
    result = metrics(y_service.loc[test_idx], service_model.predict(X.loc[test_idx]),
                     y_late.loc[test_idx], late_model.predict_proba(X.loc[test_idx])[:, 1])
    result["n_train"] = int(len(train_idx))
    return result, service_model, late_model


print("Training on the random split (80/20, stratified on late)...")
rand_train, rand_test = random_split(df)
random_metrics, service_model, late_model = fit_and_score(rand_train, rand_test)

print(f"Training on the time-based split (test: order_date >= {TIME_CUTOFF})...")
time_metrics, _, _ = fit_and_score(*time_split(df))

# Side by side with the Phase 0 baseline (original 12 features, same models)
baseline = json.loads((REPORTS / "baseline_metrics.json").read_text())
rows = []
for split, new, base in [("random", random_metrics, baseline["random_split"]),
                         ("time-based", time_metrics, baseline["time_based"])]:
    for k in ["service_mae", "service_rmse", "late_roc_auc", "late_log_loss", "late_brier", "late_share_exact_0"]:
        rows.append({"split": split, "metric": k, "baseline": base[k], "phase6": new[k],
                     "change": round(new[k] - base[k], 4)})
table = pd.DataFrame(rows)
print("\nBaseline (12 features) vs Phase 6 (17 features), same Random Forests:")
print(table.to_string(index=False))

REPORTS.mkdir(parents=True, exist_ok=True)
METRICS_PATH.write_text(json.dumps({
    "description": "Phase 6: dropped-column bug fixed (17 explicit features); same Random Forests as the baseline",
    "features": FEATURES,
    "random_split": random_metrics,
    "time_based": {"cutoff_order_date": TIME_CUTOFF, **time_metrics},
    "vs_baseline": rows,
}, indent=2))
print(f"\nSaved: {METRICS_PATH}")

# Models fitted on the random-split training part, as in the original script.
# outputs/submission_task1.csv is regenerated in Phase 9, not here.
joblib.dump(service_model, r"task1\service_model.joblib")
joblib.dump(late_model, r"task1\late_model.joblib")

print("\nSaved models:")
print("task1/service_model.joblib")
print("task1/late_model.joblib")
