"""Task 2A inference: predict_task2a(test_inputs_df) -> row_id, pred_total_volume_m3, pred_chilled_volume_m3.

Loads the saved fitted models (task2a/models/task2a_model.joblib); no retraining.
Daily calendar features for the requested weeks come from task2a/data/
(written by task2a/prepare_features.py from calendar.csv and outlets.csv).
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast import Context, predict_config  # noqa: E402
from models import WEEK_KEYS  # noqa: E402

DATA = Path("task2a") / "data"
MODEL_PATH = Path("task2a") / "models" / "task2a_model.joblib"
OUTPUT_COLUMNS = ["row_id", "pred_total_volume_m3", "pred_chilled_volume_m3"]


def load_bundle(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def predict_task2a(test_inputs_df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """test_inputs_df: row_id, depot, brand, iso_year, iso_week (one row per depot x brand x week).
    Returns one row per input row, in the input order."""
    bundle = bundle or load_bundle()
    inputs = test_inputs_df[["row_id"] + WEEK_KEYS].copy()
    assert inputs["row_id"].is_unique and not inputs.duplicated(WEEK_KEYS).any()

    weeks = set(zip(inputs["iso_year"], inputs["iso_week"]))
    daily = pd.read_csv(DATA / "daily_features.csv", parse_dates=["date"])
    outlets = pd.read_csv(DATA / "style_outlet_days.csv", parse_dates=["date"])
    in_weeks = lambda df: df[[k in weeks for k in zip(df["iso_year"], df["iso_week"])]]  # noqa: E731
    daily_test, outlet_test = in_weeks(daily), in_weeks(outlets)
    assert (daily_test["date"] > pd.Timestamp(bundle["history_end"])).all(), "forecast weeks overlap the training history"
    assert (daily_test.groupby(WEEK_KEYS).size() == 7).all() and len(daily_test) == 7 * len(
        daily_test[WEEK_KEYS].drop_duplicates()), "incomplete feature weeks"

    # Every depot x brand series is forecast for the requested weeks; output is filtered to the inputs
    target_weeks = daily_test[WEEK_KEYS].drop_duplicates().reset_index(drop=True)
    ctx = Context(daily_train=None, daily_test=daily_test, weekly_train=None, target_weeks=target_weeks,
                  outlet_train=None, outlet_test=outlet_test)
    pred = predict_config(bundle["fitted"], ctx)
    out = inputs.merge(pred, on=WEEK_KEYS, how="left", validate="one_to_one")
    assert out[["total", "chilled"]].notna().all().all(), "missing prediction for an input row"
    out = out.rename(columns={"total": "pred_total_volume_m3", "chilled": "pred_chilled_volume_m3"})
    return out[OUTPUT_COLUMNS]
