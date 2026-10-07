import pandas as pd
import joblib

from features import FEATURES
from models import CLIP_HI, CLIP_LO, predict_late

# Load trained models (fitted on all training data by task1/train_model.py)
service_model = joblib.load(r"task1\service_model.joblib")
late_model = joblib.load(r"task1\late_model.joblib")

# Load prepared test features
test = pd.read_csv(r"task1\test_features.csv")

# Phase 9 fix: select the model's features explicitly. test_features.csv also holds
# the Phase 7 candidate columns that were not kept.
missing = [c for c in FEATURES if c not in test.columns]
assert not missing, f"feature columns missing from test_features.csv: {missing}"
X_test = test[FEATURES]

# Predict
service_pred = service_model.predict(X_test)
late_prob = predict_late(late_model, X_test)  # clipped to [0.001, 0.999]

# Safety limits
service_pred = service_pred.clip(min=0)

predictions = pd.DataFrame({
    "delivery_id": test["delivery_id"],
    "pred_service_min": service_pred,
    "pred_late_prob": late_prob
})

# Load official template
template = pd.read_csv(
    r"Submission Templates\submission_task1.csv"
)

# Validate IDs
assert len(template) == len(predictions), "Row count mismatch"
assert template["delivery_id"].is_unique, "Duplicate delivery_id in template"
assert set(template["delivery_id"]) == set(predictions["delivery_id"]), \
    "delivery_id mismatch"

# Preserve EXACT template row order
service_map = predictions.set_index("delivery_id")["pred_service_min"]
late_map = predictions.set_index("delivery_id")["pred_late_prob"]

submission = template[["delivery_id"]].copy()
submission["pred_service_min"] = submission["delivery_id"].map(service_map)
submission["pred_late_prob"] = submission["delivery_id"].map(late_map)

# Final checks
assert len(submission) == 5014
assert submission["delivery_id"].tolist() == template["delivery_id"].tolist(), "delivery_id order differs"
assert list(submission.columns) == list(template.columns)
assert submission[["pred_service_min", "pred_late_prob"]].notna().all().all()
assert (submission["pred_service_min"] >= 0).all()
assert submission["pred_late_prob"].between(CLIP_LO, CLIP_HI).all()

submission.to_csv(
    r"outputs\submission_task1.csv",
    index=False
)

print("Task 1 submission created")
print("Rows:", len(submission))
print("Missing predictions:", submission[
    ["pred_service_min", "pred_late_prob"]
].isna().sum().sum())
print(f"Mean service {submission['pred_service_min'].mean():.2f} min, mean late probability "
      f"{submission['pred_late_prob'].mean():.4f}")
print("Saved: outputs/submission_task1.csv")
