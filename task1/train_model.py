import json
from pathlib import Path

import pandas as pd
import joblib

from evaluation import TIME_CUTOFF, metrics, random_split, time_split
from features import FEATURES, assert_no_leakage
from models import make_rf_models

REPORTS = Path("task1") / "reports"
METRICS_PATH = REPORTS / "metrics.json"  # latest run; phase records are metrics_phase<N>.json

df = pd.read_csv(r"task1\train_features.csv")

# Explicit features: fail loudly if any is missing (Phase 6 fix for silently dropped columns)
missing = [c for c in FEATURES if c not in df.columns]
assert not missing, f"feature columns missing from train_features.csv: {missing}"

assert_no_leakage(FEATURES)
X = df[FEATURES]
y_service = df["service_minutes"]
y_late = df["late"]


def fit_and_score(train_idx, test_idx):
    service_model, late_model = make_rf_models(FEATURES)
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
        rows.append({"split": split, "metric": k, "baseline": base[k], "current": new[k],
                     "change": round(new[k] - base[k], 4)})
table = pd.DataFrame(rows)
print(f"\nBaseline (12 features) vs current ({len(FEATURES)} features), same Random Forests:")
print(table.to_string(index=False))

REPORTS.mkdir(parents=True, exist_ok=True)
METRICS_PATH.write_text(json.dumps({
    "description": f"Random Forests as the baseline, {len(FEATURES)} explicit features (task1/features.py)",
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
