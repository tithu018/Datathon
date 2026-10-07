import pandas as pd
import joblib

# Load trained models
service_model = joblib.load(r"task1\service_model.joblib")
late_model = joblib.load(r"task1\late_model.joblib")

# Load prepared test features
test = pd.read_csv(r"task1\test_features.csv")

# Keep ID separately
X_test = test.drop(columns=["delivery_id"])

# Predict
service_pred = service_model.predict(X_test)
late_prob = late_model.predict_proba(X_test)[:, 1]

# Safety limits
service_pred = service_pred.clip(min=0)
late_prob = late_prob.clip(0, 1)

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

template["pred_service_min"] = template["delivery_id"].map(service_map)
template["pred_late_prob"] = template["delivery_id"].map(late_map)

# Final checks
assert template["pred_service_min"].notna().all()
assert template["pred_late_prob"].notna().all()
assert template["pred_late_prob"].between(0, 1).all()
assert (template["pred_service_min"] >= 0).all()

template.to_csv(
    r"outputs\submission_task1.csv",
    index=False
)

print("Task 1 submission created")
print("Rows:", len(template))
print("Missing predictions:", template[
    ["pred_service_min", "pred_late_prob"]
].isna().sum().sum())
print("Saved: outputs/submission_task1.csv")