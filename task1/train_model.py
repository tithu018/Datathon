import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, roc_auc_score

df = pd.read_csv(r"task1\train_features.csv")

X = df.drop(columns=["service_minutes", "late"])
y_service = df["service_minutes"]
y_late = df["late"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numeric_cols,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_cols,
        ),
    ]
)

X_train, X_val, ys_train, ys_val, yl_train, yl_val = train_test_split(
    X,
    y_service,
    y_late,
    test_size=0.2,
    random_state=42,
    stratify=y_late
)

service_model = Pipeline([
    ("prep", preprocessor),
    (
        "model",
        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2
        )
    )
])

late_model = Pipeline([
    ("prep", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
            class_weight="balanced"
        )
    )
])

print("Training service-time model...")
service_model.fit(X_train, ys_train)

print("Training lateness model...")
late_model.fit(X_train, yl_train)

service_pred = service_model.predict(X_val)
late_prob = late_model.predict_proba(X_val)[:, 1]

mae = mean_absolute_error(ys_val, service_pred)
auc = roc_auc_score(yl_val, late_prob)

print("\nValidation results")
print("Service MAE:", round(mae, 3))
print("Late ROC-AUC:", round(auc, 4))

joblib.dump(service_model, r"task1\service_model.joblib")
joblib.dump(late_model, r"task1\late_model.joblib")

print("\nSaved models:")
print("task1/service_model.joblib")
print("task1/late_model.joblib")