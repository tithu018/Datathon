"""Task 1 evaluation helpers: the random split used by the original pipeline, the
time-based test (train on order_date < 2026-01-03, test on the last 6 weeks of
training orders) and the metrics recorded in task1/reports/."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
TIME_CUTOFF = "2026-01-03"


def random_split(df: pd.DataFrame):
    """Same call as the original train_model.py: 80/20, stratified on late, random_state 42."""
    return train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=df["late"])


def time_split(df: pd.DataFrame):
    test = pd.to_datetime(df["order_date"]) >= pd.Timestamp(TIME_CUTOFF)
    return df.index[~test], df.index[test]


def metrics(y_service, service_pred, y_late, late_prob) -> dict:
    p = np.asarray(late_prob)
    return {
        "n_test": int(len(y_service)),
        "test_late_rate": round(float(np.mean(y_late)), 4),
        "service_mae": round(float(mean_absolute_error(y_service, service_pred)), 4),
        "service_rmse": round(float(np.sqrt(mean_squared_error(y_service, service_pred))), 4),
        "late_roc_auc": round(float(roc_auc_score(y_late, p)), 4),
        "late_log_loss": round(float(log_loss(y_late, np.clip(p, 1e-15, 1 - 1e-15))), 4),
        "late_brier": round(float(brier_score_loss(y_late, p)), 4),
        "late_share_exact_0": round(float((p == 0).mean()), 4),
        "late_share_exact_1": round(float((p == 1).mean()), 4),
    }
