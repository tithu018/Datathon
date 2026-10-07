"""Task 1 models: the original pipeline's preprocessing and Random Forests (eaf7297),
built for an explicit feature list, plus the Phase 8 HistGradientBoosting candidates."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from evaluation import SEED
from features import CATEGORICAL


def make_preprocessor(features):
    categorical_cols = [c for c in features if c in CATEGORICAL]
    numeric_cols = [c for c in features if c not in CATEGORICAL]
    return ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])


def make_rf_models(features):
    """Service-time regressor and lateness classifier with the original hyperparameters."""
    service_model = Pipeline([
        ("prep", make_preprocessor(features)),
        ("model", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, min_samples_leaf=2)),
    ])
    late_model = Pipeline([
        ("prep", make_preprocessor(features)),
        ("model", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, min_samples_leaf=2,
                                         class_weight="balanced")),
    ])
    return service_model, late_model


# ---------------------------------------------------------------- Phase 8: HistGradientBoosting
def make_hgb_preprocessor(features):
    """Categoricals ordinal-encoded (unknown/missing -> NaN) for HGB's native categorical support;
    numeric columns passed through. Output columns: categoricals first, then numerics."""
    categorical_cols = [c for c in features if c in CATEGORICAL]
    numeric_cols = [c for c in features if c not in CATEGORICAL]
    prep = ColumnTransformer(transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ])
    return prep, [True] * len(categorical_cols) + [False] * len(numeric_cols)


HGB_DEFAULTS = dict(max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
                    random_state=SEED)


def make_hgb_regressor(features, loss="squared_error", **params):
    prep, cat_mask = make_hgb_preprocessor(features)
    return Pipeline([("prep", prep), ("model", HistGradientBoostingRegressor(
        loss=loss, categorical_features=cat_mask, **{**HGB_DEFAULTS, **params}))])


def make_hgb_classifier(features, **params):
    prep, cat_mask = make_hgb_preprocessor(features)
    return Pipeline([("prep", prep), ("model", HistGradientBoostingClassifier(
        categorical_features=cat_mask, **{**HGB_DEFAULTS, **params}))])


def make_rf_classifier(features, class_weight="balanced"):
    return Pipeline([
        ("prep", make_preprocessor(features)),
        ("model", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, min_samples_leaf=2,
                                         class_weight=class_weight)),
    ])


# ---------------------------------------------------------------- final configuration (Phase 8 decision)
CLIP_LO, CLIP_HI = 0.001, 0.999  # fixed: guarantees finite log loss, costs almost nothing with sigmoid
FINAL_HGB_PARAMS = dict(learning_rate=0.05, max_iter=600)


def make_final_models(features):
    """Service: HGB regressor (squared_error). Lateness: HGB classifier + sigmoid calibration,
    CalibratedClassifierCV over 5 stratified, shuffled folds (every fold spans all seasons)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold
    service_model = make_hgb_regressor(features, "squared_error", **FINAL_HGB_PARAMS)
    late_model = CalibratedClassifierCV(make_hgb_classifier(features, **FINAL_HGB_PARAMS), method="sigmoid",
                                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    return service_model, late_model


def predict_late(late_model, X):
    return np.clip(late_model.predict_proba(X)[:, 1], CLIP_LO, CLIP_HI)
