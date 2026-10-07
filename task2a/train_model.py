"""Task 2A final fit: fit the approved configuration (task2a/reports/phase4_config.json)
on all history (2024-01-01 to 2026-03-29) and save the fitted models.

Run from the repo root after task2a/prepare_features.py. Writes
task2a/models/task2a_model.joblib (gitignored) and task2a/models/task2a_model_summary.json.
"""
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast import Context, fit_config  # noqa: E402
from models import SEED, WEEK_KEYS  # noqa: E402

DATA = Path("task2a") / "data"
CONFIG_PATH = Path("task2a") / "reports" / "phase4_config.json"
MODEL_DIR = Path("task2a") / "models"
MODEL_PATH = MODEL_DIR / "task2a_model.joblib"

np.random.seed(SEED)  # no model draws from numpy's global state, but set it anyway
config = json.loads(CONFIG_PATH.read_text())["config"]

daily = pd.read_csv(DATA / "daily_features.csv", parse_dates=["date"])
weekly = pd.read_csv(DATA / "weekly_volume.csv")
outlets = pd.read_csv(DATA / "style_outlet_days.csv", parse_dates=["date"])
daily_hist = daily[daily["split"] == "history"]
outlet_hist = outlets[outlets["split"] == "history"]
history_end = daily_hist["date"].max()
assert history_end == pd.Timestamp("2026-03-29"), history_end
assert len(weekly) == 702 and weekly["total_volume"].notna().all()

empty = pd.DataFrame(columns=WEEK_KEYS)
ctx = Context(daily_train=daily_hist, daily_test=daily.iloc[0:0], weekly_train=weekly, target_weeks=empty,
              outlet_train=outlet_hist, outlet_test=outlets.iloc[0:0])
fitted = fit_config(config, ctx)

bundle = {
    "fitted": fitted,
    "config": config,
    "history_end": str(history_end.date()),
    "trained_on": {"daily_rows": len(daily_hist), "weekly_rows": len(weekly), "style_outlet_rows": len(outlet_hist)},
    "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                 "scikit-learn": sklearn.__version__},
}
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, MODEL_PATH)

# Human-readable summary of the fitted parameters
fresh = fitted["brands"]["Fresh"]
glm_part = next(p for p, _ in fresh["parts"] if p["spec"]["model"] == "glm")
summary = {
    "config": config,
    "history_end": bundle["history_end"],
    "versions": bundle["versions"],
    "fresh_blend_weights": {p["spec"]["model"]: w for p, w in fresh["parts"]},
    "fresh_glm_multipliers": {f"{d} {b}": m.multipliers().round(4).to_dict() for (d, b), m in glm_part["models"].items()},
    "hgb_n_iter": int(next(p for p, _ in fresh["parts"] if p["spec"]["model"] == "hgb")["model"].model.n_iter_),
    "style_outlet_multipliers": fitted["brands"]["Style"]["model"].multipliers().round(4).to_dict(),
    "tech": {d: {"level_per_capacity_unit": round(v["level"], 4), "weekday_weights": v["weights"].round(4).to_dict()}
             for d, v in fitted["brands"]["Tech"]["depots"].items()},
    "chilled_share_depot_month": {f"{d} {m}": round(v, 4) for (d, m), v in fresh["share"].items()},
}
(MODEL_DIR / "task2a_model_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"Fitted config on history to {bundle['history_end']} "
      f"({len(daily_hist):,} daily rows, {len(outlet_hist):,} Style outlet rows)")
print("Fresh blend:", summary["fresh_blend_weights"], "| HGB iterations:", summary["hgb_n_iter"])
print("Tech level per weekday-capacity unit:", {d: v["level_per_capacity_unit"] for d, v in summary["tech"].items()})
print("Chilled share Apr-Jun:", {k: v for k, v in summary["chilled_share_depot_month"].items() if k.split()[-1] in ("4", "5", "6")})
print(f"Saved: {MODEL_PATH}, {MODEL_DIR / 'task2a_model_summary.json'}")
