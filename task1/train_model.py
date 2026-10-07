"""Task 1 final training (Phase 9).

1. Validation: the original baseline (12 features, original Random Forests) and the final
   configuration (30 features, HGB service model, HGB + sigmoid lateness model clipped to
   [0.001, 0.999]) on both time-based test periods:
     A  order_date >= 2026-01-03 (train on everything before)
     B  2025-02-16..2025-03-28, season-matched to the real test (train on everything before)
2. Final fit of both models on ALL training data; saved to task1/service_model.joblib and
   task1/late_model.joblib (gitignored).

Run from the repo root after task1/prepare_features.py. Writes task1/reports/metrics.json.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evaluation import metrics
from features import FEATURES, assert_no_leakage
from models import CLIP_HI, CLIP_LO, make_final_models, make_rf_models, predict_late

REPORTS = Path("task1") / "reports"
METRICS_PATH = REPORTS / "metrics.json"
ORIGINAL_FEATURES = ["temp_requirement", "order_units", "order_weight_kg", "order_volume_m3", "distance_km",
                     "planned_travel_duration_min", "planned_arrival_min", "planned_depart_min",
                     "window_open_min_feature", "window_close_min_feature", "monsoon", "dow"]

df = pd.read_csv(r"task1\train_features.csv")

# Explicit features: fail loudly if any is missing (Phase 6 fix for silently dropped columns)
missing = [c for c in FEATURES if c not in df.columns]
assert not missing, f"feature columns missing from train_features.csv: {missing}"
assert_no_leakage(FEATURES)

y_service = df["service_minutes"]
y_late = df["late"]
od = pd.to_datetime(df["order_date"])
PERIODS = {
    "A (2026-01-03..02-14)": (od < "2026-01-03", od >= "2026-01-03"),
    "B (2025-02-16..03-28)": (od < "2025-02-16", (od >= "2025-02-16") & (od <= "2025-03-28")),
}

results = {}
if "--no-eval" not in sys.argv:
    for name, (tr, te) in PERIODS.items():
        results[name] = {}
        # Original pipeline: 12 surviving features, Random Forests (balanced classifier), no clipping
        s, l = make_rf_models(ORIGINAL_FEATURES)
        s.fit(df.loc[tr, ORIGINAL_FEATURES], y_service[tr])
        l.fit(df.loc[tr, ORIGINAL_FEATURES], y_late[tr])
        results[name]["baseline"] = metrics(y_service[te], s.predict(df.loc[te, ORIGINAL_FEATURES]),
                                            y_late[te], l.predict_proba(df.loc[te, ORIGINAL_FEATURES])[:, 1])
        # Final configuration
        s, l = make_final_models(FEATURES)
        s.fit(df.loc[tr, FEATURES], y_service[tr])
        l.fit(df.loc[tr, FEATURES], y_late[tr])
        p = predict_late(l, df.loc[te, FEATURES])
        results[name]["final"] = {**metrics(y_service[te], s.predict(df.loc[te, FEATURES]), y_late[te], p),
                                  "late_mean_pred": round(float(p.mean()), 4)}
        print(f"validated {name}", flush=True)

    keys = ["service_mae", "service_rmse", "late_roc_auc", "late_log_loss", "late_brier", "late_share_exact_0"]
    table = pd.DataFrame([{"period": n, "metric": k, "baseline": r["baseline"][k], "final": r["final"][k]}
                          for n, r in results.items() for k in keys])
    print("\nBaseline (original pipeline) vs final configuration:")
    print(table.to_string(index=False))

# Final fit on ALL training data
service_model, late_model = make_final_models(FEATURES)
service_model.fit(df[FEATURES], y_service)
late_model.fit(df[FEATURES], y_late)
joblib.dump(service_model, r"task1\service_model.joblib")
joblib.dump(late_model, r"task1\late_model.joblib")
fitted_late = predict_late(late_model, df[FEATURES])
print(f"\nFinal models fitted on all {len(df):,} training rows "
      f"(in-sample mean late probability {fitted_late.mean():.4f} vs actual {y_late.mean():.4f})")

if results:
    REPORTS.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps({
        "description": "Phase 9 final configuration: HGB service model; HGB + sigmoid (cv=5 stratified, shuffled) "
                       f"lateness model clipped to [{CLIP_LO}, {CLIP_HI}]; final models fitted on all training data",
        "features": FEATURES,
        "periods": {n: {"n_train": int(tr.sum()), "n_test": int(te.sum())} for n, (tr, te) in PERIODS.items()},
        "results": results,
    }, indent=2))
    print(f"Saved: {METRICS_PATH}")

print("\nSaved models:")
print("task1/service_model.joblib")
print("task1/late_model.joblib")
